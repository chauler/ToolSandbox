"""Wrapper that attaches a :class:`~tool_sandbox.roles.tool_filter.ToolFilter`
to any existing agent, making it evaluatable as a first-class agent type in
the ToolSandbox benchmark.

Usage
-----
.. code-block:: python

    from tool_sandbox.roles.openai_api_agent import GPT_4_o_2024_05_13_Agent
    from tool_sandbox.roles.tool_filter import KeywordToolFilter
    from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent

    inner = GPT_4_o_2024_05_13_Agent()
    filt  = KeywordToolFilter(
        keyword_to_tools={
            r"(message|send|text)":  ["send_message_with_phone_number", "search_messages"],
            r"(contact|person)":     ["search_contacts", "add_contact", "modify_contact", "remove_contact"],
            r"(remind|alarm|timer)": ["add_reminder", "modify_reminder", "remove_reminder", "search_reminder"],
        },
        default_tools=["end_conversation"],
    )
    agent = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
    # ``agent`` can now be used anywhere a BaseRole is expected.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.tool_filter import ToolFilter


class ToolFilteredAgent(BaseRole):
    """Thin wrapper that attaches a :class:`ToolFilter` to an existing agent.

    The wrapper delegates *all* behaviour to the ``inner_agent``.  The only
    change is that the inner agent's ``_tool_filter`` attribute is set so that
    :meth:`BaseRole.get_available_tools` applies the filter before returning
    tools to the model.

    This means:
    * The inner agent's ``respond()``, ``reset()``, and ``teardown()`` are used
      directly – no message interception or monkey-patching needed.
    * The wrapper is suitable for use in the ``roles`` dict passed to
      :meth:`Scenario.play`.
    * Evaluation code sees exactly the same trajectory format as for an
      unfiltered agent.

    Args:
        inner_agent: The agent instance to wrap (e.g. ``GPT_4_o_2024_05_13_Agent()``).
        tool_filter: The filter to apply each time the agent requests tools.
    """

    role_type: RoleType = RoleType.AGENT

    def __init__(
        self,
        inner_agent: BaseRole,
        tool_filter: ToolFilter,
    ) -> None:
        self.inner_agent = inner_agent
        self.tool_filter = tool_filter
        # Attach the filter directly to the inner agent so its own
        # ``get_available_tools()`` call (inside ``respond()``) picks it up.
        self.inner_agent._tool_filter = tool_filter

    # --- Properties forwarded from the inner agent ---

    @property
    def model_name(self) -> str:  # type: ignore[override]
        """Expose the inner model name for logging / output directory naming."""
        base_name = getattr(self.inner_agent, "model_name", "unknown")
        filter_tag = type(self.tool_filter).__name__
        return f"{base_name}_filtered_{filter_tag}"

    # --- Delegate lifecycle methods ---

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Forward to the inner agent's respond."""
        self.inner_agent.respond(ending_index=ending_index)

    def reset(self) -> None:
        """Forward to the inner agent's reset."""
        self.inner_agent.reset()

    def teardown(self) -> None:
        """Forward to the inner agent's teardown."""
        self.inner_agent.teardown()

    def get_available_tools(self) -> dict[str, Callable[..., Any]]:
        """Return filtered tools.

        Normally the inner agent's own ``get_available_tools`` is sufficient
        (because we set ``_tool_filter`` on it).  This override ensures that
        callers who hold a reference to the *wrapper* also get filtered tools.
        """
        return self.inner_agent.get_available_tools()
