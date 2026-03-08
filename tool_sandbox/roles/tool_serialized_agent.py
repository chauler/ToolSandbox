"""Wrapper that attaches a :class:`~tool_sandbox.roles.tool_serializer.ToolSerializer`
to any existing agent, making it evaluatable as a first-class agent type in
the ToolSandbox benchmark.

Usage
-----
.. code-block:: python

    from tool_sandbox.roles.openai_api_agent import GPT_4_o_2024_05_13_Agent
    from tool_sandbox.roles.tool_serializer import CompactDescriptionSerializer
    from tool_sandbox.roles.tool_serialized_agent import ToolSerializedAgent

    inner = GPT_4_o_2024_05_13_Agent()
    ser   = CompactDescriptionSerializer(max_tool_desc_length=40)
    agent = ToolSerializedAgent(inner_agent=inner, tool_serializer=ser)
    # ``agent`` can now be used anywhere a BaseRole is expected.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.tool_serializer import ToolSerializer


class ToolSerializedAgent(BaseRole):
    """Thin wrapper that attaches a :class:`ToolSerializer` to an existing agent.

    The wrapper delegates *all* behaviour to the ``inner_agent``.  The only
    change is that the inner agent's ``_tool_serializer`` attribute is set so
    that :meth:`BaseRole.serialize_tools` applies the serializer after the
    standard ``convert_to_openai_tools`` conversion.

    This means:

    * The inner agent's ``respond()``, ``reset()``, and ``teardown()`` are used
      directly – no message interception or monkey-patching needed.
    * The wrapper is suitable for use in the ``roles`` dict passed to
      :meth:`Scenario.play`.
    * Evaluation code sees exactly the same trajectory format as for an
      unserialized agent.

    Args:
        inner_agent:     The agent instance to wrap.
        tool_serializer: The serializer to apply each time tools are converted.
    """

    role_type: RoleType = RoleType.AGENT

    def __init__(
        self,
        inner_agent: BaseRole,
        tool_serializer: ToolSerializer,
    ) -> None:
        self.inner_agent = inner_agent
        self.tool_serializer = tool_serializer
        # Attach the serializer directly to the inner agent so its own
        # ``serialize_tools()`` call (inside ``respond()``) picks it up.
        self.inner_agent._tool_serializer = tool_serializer

    # --- Properties forwarded from the inner agent ---

    @property
    def model_name(self) -> str:  # type: ignore[override]
        """Expose the inner model name for logging / output directory naming."""
        base_name = getattr(self.inner_agent, "model_name", "unknown")
        serializer_tag = type(self.tool_serializer).__name__
        return f"{base_name}_serialized_{serializer_tag}"

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
        """Return tools from the inner agent.

        Normally the inner agent's own ``get_available_tools`` is sufficient.
        This override ensures that callers who hold a reference to the *wrapper*
        also get the same tools.
        """
        return self.inner_agent.get_available_tools()

    def serialize_tools(
        self, tools: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Return serialized tools from the inner agent."""
        return self.inner_agent.serialize_tools(tools)
