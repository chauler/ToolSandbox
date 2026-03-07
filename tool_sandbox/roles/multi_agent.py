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
