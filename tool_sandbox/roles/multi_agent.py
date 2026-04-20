"""Multi-agent architecture allowing multiple agents to collaborate on tasks.

A :class:`MultiAgentRole` wraps several sub-agents and a :class:`AgentRouter`
that decides which sub-agent handles each conversational turn.  From the
ToolSandbox benchmark's perspective the whole ensemble appears as a single
``AGENT`` role, so the evaluation pipeline works without modification.

Usage
-----
.. code-block:: python

    from tool_sandbox.roles.openai_api_agent import GPT_4_o_2024_05_13_Agent
    from tool_sandbox.roles.anthropic_api_agent import ClaudeHaikuAgent
    from tool_sandbox.roles.multi_agent import MultiAgentRole, LLMRouter

    planner = GPT_4_o_2024_05_13_Agent()
    executor = ClaudeHaikuAgent()
    router  = LLMRouter(model_name="gpt-4o-mini")
    multi   = MultiAgentRole(
        agents={"planner": planner, "executor": executor},
        router=router,
    )
    # ``multi`` can be used anywhere a single agent BaseRole is expected.
"""

from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, Optional, Sequence

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

LOGGER = getLogger(__name__)


# ---------------------------------------------------------------------------
# Router abstraction
# ---------------------------------------------------------------------------


class AgentRouter(ABC):
    """Decides which sub-agent handles the current conversational turn."""

    @abstractmethod
    def route(
        self,
        messages: list[Message],
        agents: dict[str, BaseRole],
    ) -> str:
        """Return the *name* (key in *agents*) of the agent that should
        handle this turn.

        Args:
            messages: Full conversation history.
            agents:   Mapping of agent name to agent instance.

        Returns:
            Name of the selected agent.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete routers
# ---------------------------------------------------------------------------


class RoundRobinRouter(AgentRouter):
    """Cycle through agents in a fixed order.

    Useful as a trivial baseline or for architectures where each agent has a
    fixed role in a predetermined sequence (e.g. plan-then-execute).

    Args:
        order: Explicit ordering.  If *None* the alphabetical ordering
               of agent names is used.
    """

    def __init__(self, order: Optional[Sequence[str]] = None) -> None:
        self._order = list(order) if order is not None else None
        self._cycle: Optional[itertools.cycle[str]] = None  # type: ignore[type-arg]

    def route(
        self,
        messages: list[Message],
        agents: dict[str, BaseRole],
    ) -> str:
        if self._cycle is None:
            ordering = self._order if self._order is not None else sorted(agents)
            self._cycle = itertools.cycle(ordering)
        return next(self._cycle)


class KeywordRouter(AgentRouter):
    """Route based on keyword matching in the most recent message.

    Args:
        keyword_to_agent: Mapping from regex pattern to agent name.
        default_agent:    Fallback agent when no pattern matches.
    """

    def __init__(
        self,
        keyword_to_agent: dict[str, str],
        default_agent: str,
    ) -> None:
        import re

        self.keyword_to_agent = {
            re.compile(pat, re.IGNORECASE): agent_name
            for pat, agent_name in keyword_to_agent.items()
        }
        self.default_agent = default_agent

    def route(
        self,
        messages: list[Message],
        agents: dict[str, BaseRole],
    ) -> str:
        last_content = messages[-1].content if messages else ""
        for pattern, agent_name in self.keyword_to_agent.items():
            if pattern.search(last_content or ""):
                return agent_name
        return self.default_agent


class LLMRouter(AgentRouter):
    """Use an LLM to route the current turn to the most appropriate agent.

    The router sends a prompt describing each available sub-agent and the
    recent conversation, and asks the LLM to pick one agent.

    Args:
        model_name:        OpenAI-compatible model name.
        api_key:           Optional API key (defaults to ``OPENAI_API_KEY``).
        base_url:          Optional base URL for the endpoint.
        agent_descriptions: Mapping from agent name to a human-readable
                            description of that agent's capabilities.  If not
                            provided, the agent class name is used.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        else:
            kwargs["base_url"] = "https://api.openai.com/v1"
        self._client = OpenAI(**kwargs)
        self.model_name = model_name
        self.agent_descriptions = agent_descriptions or {}

    def route(
        self,
        messages: list[Message],
        agents: dict[str, BaseRole],
    ) -> str:
        agent_desc_parts: list[str] = []
        for name, agent in agents.items():
            desc = self.agent_descriptions.get(
                name,
                f"{type(agent).__name__} (model: {getattr(agent, 'model_name', 'n/a')})",
            )
            agent_desc_parts.append(f"- {name}: {desc}")
        agents_str = "\n".join(agent_desc_parts)

        recent = messages[-5:] if len(messages) > 5 else messages
        context_parts: list[str] = []
        for msg in recent:
            sender = msg.sender.value if msg.sender else "unknown"
            content = (msg.content or "")[:500]
            context_parts.append(f"[{sender}]: {content}")
        context_str = "\n".join(context_parts)

        prompt = (
            "You are a routing controller for a multi-agent system. Based on "
            "the conversation context and the available agents below, respond "
            "with ONLY the name of the agent that should handle the current "
            "turn. Do not include any other text.\n\n"
            f"Available agents:\n{agents_str}\n\n"
            f"Conversation context:\n{context_str}\n\n"
            "Selected agent name:"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            chosen = (response.choices[0].message.content or "").strip()
            # Fuzzy match against known names
            if chosen in agents:
                return chosen
            # Try case-insensitive match
            for name in agents:
                if name.lower() == chosen.lower():
                    return name
            LOGGER.warning(
                "LLM router returned unknown agent '%s'; falling back to first agent.",
                chosen,
            )
        except Exception:
            LOGGER.warning("LLM router failed; falling back to first agent.", exc_info=True)

        return next(iter(agents))


class ToolBasedRouter(AgentRouter):
    """Route based on which agent has the tools likely needed for this turn.

    Analyses the most recent user message against each sub-agent's available
    tools (via simple keyword overlap on tool names and docstrings) and picks
    the agent with the best coverage.

    Args:
        default_agent: Fallback agent when scores are tied or zero.
    """

    def __init__(self, default_agent: Optional[str] = None) -> None:
        self.default_agent = default_agent

    def route(
        self,
        messages: list[Message],
        agents: dict[str, BaseRole],
    ) -> str:
        last_content = (messages[-1].content or "").lower() if messages else ""
        words = set(last_content.split())

        scores: dict[str, int] = {}
        for name, agent in agents.items():
            try:
                agent_tools = agent.get_available_tools()
            except Exception:
                agent_tools = {}
            score = 0
            for tool_name, tool_fn in agent_tools.items():
                tool_words = set(tool_name.lower().replace("_", " ").split())
                doc_words = set(
                    (getattr(tool_fn, "__doc__", "") or "").lower().split()[:30]
                )
                overlap = words & (tool_words | doc_words)
                score += len(overlap)
            scores[name] = score

        if not scores or max(scores.values()) == 0:
            if self.default_agent and self.default_agent in agents:
                return self.default_agent
            return next(iter(agents))

        return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Multi-agent role
# ---------------------------------------------------------------------------

class _MutableAllowListToolFilter:
    """Private mutable allow-list filter used by SK tool selection."""

    def __init__(self, allowed_tool_names: Optional[Sequence[str]] = None) -> None:
        self.allowed_tool_names: set[str] = set(allowed_tool_names or [])

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        _ = messages
        if not self.allowed_tool_names:
            return tools
        return {
            name: tool
            for name, tool in tools.items()
            if name in self.allowed_tool_names
        }


class SemanticKernelToolSelectorAgent:
    """Selector sub-agent that uses Semantic Kernel to rank tools."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        top_fraction: float = 0.3,
        min_tools: int = 1,
        max_tools: Optional[int] = None,
        always_include: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_to_all: bool = True,
    ) -> None:
        if top_fraction <= 0 or top_fraction > 1:
            raise ValueError("top_fraction must be in (0, 1].")
        if min_tools < 1:
            raise ValueError("min_tools must be >= 1.")
        self.model_name = model_name
        self.top_fraction = top_fraction
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.always_include: set[str] = set(always_include or [])
        self.api_key = api_key
        self.base_url = base_url
        self.fallback_to_all = fallback_to_all
        self._kernel = None

    def _init_kernel(self) -> Any:
        if self._kernel is not None:
            return self._kernel

        try:
            import importlib

            sk_module = importlib.import_module("semantic_kernel")
            sk_openai_module = importlib.import_module(
                "semantic_kernel.connectors.ai.open_ai"
            )
            openai_module = importlib.import_module("openai")
            Kernel = getattr(sk_module, "Kernel")
            OpenAIChatCompletion = getattr(sk_openai_module, "OpenAIChatCompletion")
            AsyncOpenAI = getattr(openai_module, "AsyncOpenAI")
        except Exception as exc:
            raise RuntimeError(
                "Semantic Kernel is required for SemanticKernelToolSelectorAgent. "
                "Install dependency: semantic-kernel and use an OpenAI package "
                "version compatible with Semantic Kernel (for this project, "
                "Python >=3.10 uses openai==1.109.1)."
            ) from exc

        kernel = Kernel()
        service_kwargs: dict[str, Any] = {
            "service_id": "tool_selector",
            "ai_model_id": self.model_name,
        }
        if self.base_url is not None:
            async_client_kwargs: dict[str, Any] = {"base_url": self.base_url}
            if self.api_key is not None:
                async_client_kwargs["api_key"] = self.api_key
            service_kwargs["async_client"] = AsyncOpenAI(**async_client_kwargs)
        elif self.api_key is not None:
            service_kwargs["api_key"] = self.api_key
        kernel.add_service(OpenAIChatCompletion(**service_kwargs))
        self._kernel = kernel
        return kernel

    @staticmethod
    def _parse_json_array(content: str) -> list[str]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON list from selector model")
        return [str(x) for x in parsed]

    def _build_prompt(
        self,
        messages: list[Message],
        tool_names: list[str],
        tool_descriptions: list[str],
        top_k: int,
    ) -> str:
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_parts: list[str] = []
        for msg in recent_messages:
            sender = msg.sender.value if msg.sender else "unknown"
            conversation_parts.append(f"[{sender}] {(msg.content or '')[:600]}")
        conversation = "\n".join(conversation_parts)

        tool_lines = [
            f"- {name}: {desc}"
            for name, desc in zip(tool_names, tool_descriptions)
        ]
        tools_str = "\n".join(tool_lines)

        return (
            "You are a tool filtering agent in a two-agent architecture. "
            "Given conversation context and available tools, select the most "
            f"relevant tool names for the execution agent. Select at most {top_k} "
            "tools. Return ONLY a JSON array of tool names.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Available tools:\n{tools_str}\n\n"
            "Selected tool names (JSON array):"
        )

    def _invoke_prompt(self, prompt: str) -> str:
        import asyncio

        kernel = self._init_kernel()

        async def _run() -> str:
            result = await kernel.invoke_prompt(prompt)
            return str(result)

        return asyncio.run(_run())

    def select_tool_names(
        self,
        messages: list[Message],
        tools: dict[str, Callable[..., Any]],
    ) -> set[str]:
        if not tools:
            return set()

        tool_names = list(tools.keys())
        tool_descriptions = [
            (getattr(tool, "__doc__", None) or "").split("\n")[0]
            for tool in tools.values()
        ]

        top_k = max(self.min_tools, int(len(tool_names) * self.top_fraction + 0.9999))
        if self.max_tools is not None:
            top_k = min(top_k, self.max_tools)
        top_k = max(1, min(top_k, len(tool_names)))

        prompt = self._build_prompt(messages, tool_names, tool_descriptions, top_k)
        try:
            raw = self._invoke_prompt(prompt)
            selected_ordered = self._parse_json_array(raw)
            selected = [name for name in selected_ordered if name in tools]
            if len(selected) < top_k:
                for name in tool_names:
                    if name not in selected:
                        selected.append(name)
                    if len(selected) >= top_k:
                        break
            selected = selected[:top_k]
            selected_set = set(selected) | {n for n in self.always_include if n in tools}
            return selected_set
        except Exception:
            LOGGER.warning(
                "Semantic Kernel tool selection failed; falling back to %s.",
                "all tools" if self.fallback_to_all else f"top {top_k} tools",
                exc_info=True,
            )
            if self.fallback_to_all:
                return set(tool_names)
            return set(tool_names[:top_k])


class SemanticKernelToolFilterMultiAgent(BaseRole):
    """Two-agent architecture: SK selector sub-agent + execution agent."""

    role_type: RoleType = RoleType.AGENT

    def __init__(
        self,
        execution_agent: BaseRole,
        selector_agent: SemanticKernelToolSelectorAgent,
    ) -> None:
        self.execution_agent = execution_agent
        self.selector_agent = selector_agent
        self._dynamic_filter = _MutableAllowListToolFilter()

        existing_filter = getattr(self.execution_agent, "_tool_filter", None)
        if existing_filter is None:
            self.execution_agent._tool_filter = self._dynamic_filter
        else:
            from tool_sandbox.roles.tool_filter import CompositeToolFilter

            self.execution_agent._tool_filter = CompositeToolFilter(
                [existing_filter, self._dynamic_filter]
            )

    @property
    def model_name(self) -> str:  # type: ignore[override]
        base_name = getattr(self.execution_agent, "model_name", "unknown")
        return (
            f"sk_tool_filter_multi_agent("
            f"selector={self.selector_agent.model_name},"
            f"executor={base_name},"
            f"top_fraction={self.selector_agent.top_fraction}"
            f")"
        )

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages = self.get_messages(ending_index=ending_index)
        available_tools = BaseRole.get_available_tools(self)
        selected_tool_names = self.selector_agent.select_tool_names(
            messages=messages,
            tools=available_tools,
        )
        self._dynamic_filter.allowed_tool_names = selected_tool_names
        LOGGER.info(
            "SemanticKernelToolFilterMultiAgent selected %d/%d tools.",
            len(selected_tool_names),
            len(available_tools),
        )
        self.execution_agent.respond(ending_index=ending_index)

    def reset(self) -> None:
        self.execution_agent.reset()

    def teardown(self) -> None:
        self.execution_agent.teardown()

    def get_available_tools(self) -> dict[str, Callable[..., Any]]:
        return self.execution_agent.get_available_tools()


class MultiAgentRole(BaseRole):
    """Orchestrates multiple sub-agents as a single ``AGENT`` role.

    From the ToolSandbox perspective this role is indistinguishable from a
    single agent:  it receives messages addressed to ``RoleType.AGENT`` and
    writes response messages from ``RoleType.AGENT``.  Internally an
    :class:`AgentRouter` selects which sub-agent handles each turn.

    Because all sub-agents share the same global
    :class:`~tool_sandbox.common.execution_context.ExecutionContext`, there is
    no need for explicit state synchronisation.

    Args:
        agents: Named sub-agents.  Each must have ``role_type == AGENT``.
        router: Strategy for selecting which sub-agent responds each turn.
    """

    role_type: RoleType = RoleType.AGENT

    def __init__(
        self,
        agents: dict[str, BaseRole],
        router: AgentRouter,
    ) -> None:
        if not agents:
            raise ValueError("At least one sub-agent is required.")
        self.agents = agents
        self.router = router

    # --- Properties ---

    @property
    def model_name(self) -> str:  # type: ignore[override]
        """Composite name for logging and output directory naming."""
        names = []
        for key, agent in self.agents.items():
            name = getattr(agent, "model_name", type(agent).__name__)
            names.append(f"{key}={name}")
        router_name = type(self.router).__name__
        return f"multi_agent({'+'.join(names)})_router_{router_name}"

    # --- Lifecycle ---

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Route the current turn to the selected sub-agent."""
        messages = self.get_messages(ending_index=ending_index)
        agent_name = self.router.route(messages, self.agents)
        selected = self.agents.get(agent_name)
        if selected is None:
            LOGGER.error(
                "Router returned unknown agent '%s'; using first agent.", agent_name
            )
            selected = next(iter(self.agents.values()))
        LOGGER.info("MultiAgentRole routing to '%s'.", agent_name)
        selected.respond(ending_index=ending_index)

    def reset(self) -> None:
        """Reset all sub-agents."""
        for agent in self.agents.values():
            agent.reset()

    def teardown(self) -> None:
        """Tear down all sub-agents."""
        for agent in self.agents.values():
            agent.teardown()

    def get_available_tools(self) -> dict[str, Callable[..., Any]]:
        """Return union of all sub-agents' tools.

        This is primarily for introspection; per-agent filtering is respected
        when each sub-agent's own ``get_available_tools`` is called during
        ``respond``.
        """
        merged: dict[str, Callable[..., Any]] = {}
        for agent in self.agents.values():
            merged.update(agent.get_available_tools())
        return merged
