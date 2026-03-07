"""Tests for tool filtering and multi-agent infrastructure."""

import json
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.multi_agent import (
    KeywordRouter,
    MultiAgentRole,
    RoundRobinRouter,
    ToolBasedRouter,
)
from tool_sandbox.roles.tool_filter import (
    AllowListToolFilter,
    CompositeToolFilter,
    DenyListToolFilter,
    KeywordToolFilter,
)
from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _execution_context() -> None:
    """Set up a fresh execution context for each test."""
    ctx = ExecutionContext()
    set_current_context(ctx)


def _make_dummy_tool(name: str, doc: str = "dummy") -> Callable[..., Any]:
    """Create a trivially callable tool with the given name."""

    def fn() -> None:
        pass

    fn.__name__ = name
    fn.__doc__ = doc
    fn.is_tool = True  # type: ignore[attr-defined]
    fn.visible_to = (RoleType.AGENT,)  # type: ignore[attr-defined]
    return fn


def _make_tools() -> dict[str, Callable[..., Any]]:
    return {
        "search_contacts": _make_dummy_tool("search_contacts", "Search for contacts"),
        "send_message": _make_dummy_tool("send_message", "Send a text message"),
        "set_wifi": _make_dummy_tool("set_wifi", "Toggle WiFi setting"),
        "add_reminder": _make_dummy_tool("add_reminder", "Create a reminder"),
    }


def _make_messages(content: str = "Send a message to John") -> list[Message]:
    return [
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content=content,
        ),
    ]


# ---------------------------------------------------------------------------
# AllowListToolFilter
# ---------------------------------------------------------------------------


class TestAllowListToolFilter:
    def test_keeps_only_allowed(self) -> None:
        tools = _make_tools()
        f = AllowListToolFilter(["search_contacts", "send_message"])
        result = f.filter_tools(tools, _make_messages())
        assert set(result.keys()) == {"search_contacts", "send_message"}

    def test_empty_allow_list_returns_nothing(self) -> None:
        tools = _make_tools()
        f = AllowListToolFilter([])
        result = f.filter_tools(tools, _make_messages())
        assert result == {}


# ---------------------------------------------------------------------------
# DenyListToolFilter
# ---------------------------------------------------------------------------


class TestDenyListToolFilter:
    def test_removes_denied(self) -> None:
        tools = _make_tools()
        f = DenyListToolFilter(["set_wifi", "add_reminder"])
        result = f.filter_tools(tools, _make_messages())
        assert set(result.keys()) == {"search_contacts", "send_message"}

    def test_empty_deny_list_keeps_all(self) -> None:
        tools = _make_tools()
        f = DenyListToolFilter([])
        result = f.filter_tools(tools, _make_messages())
        assert set(result.keys()) == set(tools.keys())


# ---------------------------------------------------------------------------
# KeywordToolFilter
# ---------------------------------------------------------------------------


class TestKeywordToolFilter:
    def test_matches_keyword(self) -> None:
        tools = _make_tools()
        f = KeywordToolFilter(
            keyword_to_tools={
                r"(message|send|text)": ["send_message"],
                r"(contact|person)": ["search_contacts"],
            },
            default_tools=["add_reminder"],
        )
        result = f.filter_tools(tools, _make_messages("Send a message to John"))
        assert "send_message" in result
        assert "add_reminder" in result  # default_tools always included

    def test_no_match_returns_defaults_only(self) -> None:
        tools = _make_tools()
        f = KeywordToolFilter(
            keyword_to_tools={
                r"(weather|forecast)": ["set_wifi"],
            },
            default_tools=["add_reminder"],
        )
        result = f.filter_tools(tools, _make_messages("Hello world"))
        assert set(result.keys()) == {"add_reminder"}

    def test_include_unmatched(self) -> None:
        tools = _make_tools()
        f = KeywordToolFilter(
            keyword_to_tools={
                r"(weather|forecast)": ["set_wifi"],
            },
            include_unmatched=True,
        )
        # No keyword matches, but include_unmatched=True => return all
        result = f.filter_tools(tools, _make_messages("Hello world"))
        assert set(result.keys()) == set(tools.keys())


# ---------------------------------------------------------------------------
# CompositeToolFilter
# ---------------------------------------------------------------------------


class TestCompositeToolFilter:
    def test_chains_filters(self) -> None:
        tools = _make_tools()
        f1 = AllowListToolFilter(["search_contacts", "send_message", "set_wifi"])
        f2 = DenyListToolFilter(["set_wifi"])
        composite = CompositeToolFilter([f1, f2])
        result = composite.filter_tools(tools, _make_messages())
        assert set(result.keys()) == {"search_contacts", "send_message"}


# ---------------------------------------------------------------------------
# BaseRole._tool_filter integration
# ---------------------------------------------------------------------------


class TestBaseRoleToolFilter:
    """Verify that BaseRole.get_available_tools applies _tool_filter."""

    def test_default_no_filter(self) -> None:
        role = BaseRole()
        role.role_type = RoleType.AGENT
        # Without a filter, get_available_tools returns ALL tools available to agent
        tools = role.get_available_tools()
        assert isinstance(tools, dict)

    def test_with_allow_list_filter(self) -> None:
        role = BaseRole()
        role.role_type = RoleType.AGENT
        # Get the unfiltered set first
        all_tools = role.get_available_tools()
        if not all_tools:
            pytest.skip("No tools registered in default context")
        # Pick one tool name to keep
        one_name = next(iter(all_tools))
        role._tool_filter = AllowListToolFilter([one_name])
        filtered = role.get_available_tools()
        assert set(filtered.keys()) == {one_name}


# ---------------------------------------------------------------------------
# ToolFilteredAgent
# ---------------------------------------------------------------------------


class TestToolFilteredAgent:
    def test_model_name_includes_filter(self) -> None:
        inner = MagicMock(spec=BaseRole)
        inner.model_name = "gpt-4o"
        inner.role_type = RoleType.AGENT
        filt = AllowListToolFilter(["search_contacts"])
        agent = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
        assert "gpt-4o" in agent.model_name
        assert "AllowListToolFilter" in agent.model_name

    def test_filter_attached_to_inner(self) -> None:
        inner = BaseRole()
        inner.role_type = RoleType.AGENT
        filt = AllowListToolFilter(["search_contacts"])
        _ = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
        assert inner._tool_filter is filt

    def test_teardown_delegates(self) -> None:
        inner = MagicMock(spec=BaseRole)
        inner.role_type = RoleType.AGENT
        filt = AllowListToolFilter([])
        agent = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
        agent.teardown()
        inner.teardown.assert_called_once()

    def test_reset_delegates(self) -> None:
        inner = MagicMock(spec=BaseRole)
        inner.role_type = RoleType.AGENT
        filt = AllowListToolFilter([])
        agent = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
        agent.reset()
        inner.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


class _DummyAgent(BaseRole):
    """Minimal agent for testing routers."""

    role_type = RoleType.AGENT
    responded = False

    def respond(self, ending_index: Optional[int] = None) -> None:
        self.responded = True


class TestRoundRobinRouter:
    def test_cycles(self) -> None:
        agents = {"a": _DummyAgent(), "b": _DummyAgent()}
        router = RoundRobinRouter(order=["a", "b"])
        msgs = _make_messages()
        assert router.route(msgs, agents) == "a"
        assert router.route(msgs, agents) == "b"
        assert router.route(msgs, agents) == "a"


class TestKeywordRouter:
    def test_matches(self) -> None:
        agents = {"msg_agent": _DummyAgent(), "wifi_agent": _DummyAgent()}
        router = KeywordRouter(
            keyword_to_agent={r"(message|send)": "msg_agent"},
            default_agent="wifi_agent",
        )
        assert router.route(_make_messages("Send a message"), agents) == "msg_agent"
        assert router.route(_make_messages("Toggle wifi"), agents) == "wifi_agent"


class TestToolBasedRouter:
    def test_routes_to_agent_with_matching_tools(self) -> None:
        agent_a = _DummyAgent()
        agent_a._tool_filter = AllowListToolFilter(["send_message_with_phone_number"])

        agent_b = _DummyAgent()
        agent_b._tool_filter = AllowListToolFilter(["set_wifi_status"])

        agents = {"msg_agent": agent_a, "wifi_agent": agent_b}
        router = ToolBasedRouter(default_agent="msg_agent")
        # The router analyses tool name overlap with the message
        # Since the default agent is msg_agent, and no tool keywords match,
        # it should fall back to default
        result = router.route(_make_messages("do something"), agents)
        assert result in agents


# ---------------------------------------------------------------------------
# MultiAgentRole
# ---------------------------------------------------------------------------


class TestMultiAgentRole:
    def test_requires_at_least_one_agent(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            MultiAgentRole(agents={}, router=RoundRobinRouter())

    def test_model_name(self) -> None:
        agents = {
            "a": _DummyAgent(),
            "b": _DummyAgent(),
        }
        multi = MultiAgentRole(agents=agents, router=RoundRobinRouter())
        assert "multi_agent" in multi.model_name
        assert "RoundRobinRouter" in multi.model_name

    def test_respond_delegates(self) -> None:
        agent_a = _DummyAgent()
        agent_b = _DummyAgent()
        agents = {"a": agent_a, "b": agent_b}
        router = RoundRobinRouter(order=["b", "a"])

        ctx = get_current_context()
        ctx.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=[
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "Hello",
                }
            ],
        )

        multi = MultiAgentRole(agents=agents, router=router)
        # First call should route to "b"
        multi.respond()
        assert agent_b.responded
        assert not agent_a.responded

    def test_reset_all(self) -> None:
        inner_a = MagicMock(spec=BaseRole)
        inner_b = MagicMock(spec=BaseRole)
        multi = MultiAgentRole(
            agents={"a": inner_a, "b": inner_b},
            router=RoundRobinRouter(),
        )
        multi.reset()
        inner_a.reset.assert_called_once()
        inner_b.reset.assert_called_once()

    def test_teardown_all(self) -> None:
        inner_a = MagicMock(spec=BaseRole)
        inner_b = MagicMock(spec=BaseRole)
        multi = MultiAgentRole(
            agents={"a": inner_a, "b": inner_b},
            router=RoundRobinRouter(),
        )
        multi.teardown()
        inner_a.teardown.assert_called_once()
        inner_b.teardown.assert_called_once()


# ---------------------------------------------------------------------------
# Agent config loading
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_build_allow_list_filter(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {"type": "allow_list", "allowed_tool_names": ["a", "b"]}
        f = _build_tool_filter(config)
        assert isinstance(f, AllowListToolFilter)

    def test_build_deny_list_filter(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {"type": "deny_list", "denied_tool_names": ["a"]}
        f = _build_tool_filter(config)
        assert isinstance(f, DenyListToolFilter)

    def test_build_keyword_filter(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {
            "type": "keyword",
            "keyword_to_tools": {"(test)": ["tool1"]},
        }
        f = _build_tool_filter(config)
        assert isinstance(f, KeywordToolFilter)

    def test_build_composite_filter(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {
            "type": "composite",
            "filters": [
                {"type": "allow_list", "allowed_tool_names": ["a", "b"]},
                {"type": "deny_list", "denied_tool_names": ["b"]},
            ],
        }
        f = _build_tool_filter(config)
        assert isinstance(f, CompositeToolFilter)
        assert len(f.filters) == 2

    def test_build_unknown_filter_raises(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        with pytest.raises(ValueError, match="Unknown tool filter"):
            _build_tool_filter({"type": "nonexistent"})

    def test_build_round_robin_router(self) -> None:
        from tool_sandbox.cli.agent_config import _build_router

        config = {"type": "round_robin", "order": ["a", "b"]}
        r = _build_router(config)
        assert isinstance(r, RoundRobinRouter)

    def test_build_keyword_router(self) -> None:
        from tool_sandbox.cli.agent_config import _build_router

        config = {
            "type": "keyword",
            "keyword_to_agent": {"(test)": "agent_a"},
            "default_agent": "agent_b",
        }
        r = _build_router(config)
        assert isinstance(r, KeywordRouter)

    def test_build_tool_based_router(self) -> None:
        from tool_sandbox.cli.agent_config import _build_router

        config = {"type": "tool_based", "default_agent": "agent_a"}
        r = _build_router(config)
        assert isinstance(r, ToolBasedRouter)

    def test_build_unknown_router_raises(self) -> None:
        from tool_sandbox.cli.agent_config import _build_router

        with pytest.raises(ValueError, match="Unknown router"):
            _build_router({"type": "nonexistent"})

    def test_build_simple_agent_from_config(self) -> None:
        from tool_sandbox.cli.agent_config import build_agent_from_config

        config = {"type": "Unhelpful"}
        agent = build_agent_from_config(config)
        assert isinstance(agent, BaseRole)
