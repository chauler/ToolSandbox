from typing import Any, Callable, Optional

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.roles.base_role import BaseRole

class AgentFrameworkAgent(BaseRole):
    role_type: RoleType = RoleType.AGENT

    def __init__(self, inner_agent: BaseRole) -> None:
        self.inner_agent = inner_agent

    @property
    def model_name(self) -> str:  # type: ignore[override]
        """Expose the inner model name for logging / output directory naming."""
        base_name = getattr(self.inner_agent, "model_name", "unknown")
        filter_tag = type(self.tool_filter).__name__
        return f"{base_name}_filtered_{filter_tag}"

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