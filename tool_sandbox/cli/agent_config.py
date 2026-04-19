"""Load agent configurations from JSON files.

This module supports constructing complex agent topologies (filtered agents,
multi-agent ensembles, etc.) from a declarative JSON specification.  The
configuration is passed via the ``--agent_config`` CLI argument.

Configuration format
--------------------

**Simple agent** (equivalent to ``--agent GPT_4_o_2024_05_13``)::

    {
        "type": "GPT_4_o_2024_05_13"
    }

**Agent with a tool filter**::

    {
        "type": "tool_filtered",
        "inner_agent": "GPT_4_o_2024_05_13",
        "tool_filter": {
            "type": "keyword",
            "keyword_to_tools": {
                "(message|send|text)": [
                    "send_message_with_phone_number",
                    "search_messages"
                ],
                "(contact|person)": [
                    "search_contacts",
                    "add_contact",
                    "modify_contact",
                    "remove_contact"
                ]
            },
            "default_tools": ["end_conversation"],
            "include_unmatched": false
        }
    }

**Agent with an LLM classifier filter**::

    {
        "type": "tool_filtered",
        "inner_agent": "Claude_3_Haiku",
        "tool_filter": {
            "type": "llm_classifier",
            "model_name": "gpt-4o-mini",
            "always_include": ["end_conversation"]
        }
    }

**Multi-agent**::

    {
        "type": "multi_agent",
        "agents": {
            "planner": {
                "type": "GPT_4_o_2024_05_13"
            },
            "executor": {
                "type": "tool_filtered",
                "inner_agent": "Claude_3_Haiku",
                "tool_filter": {
                    "type": "allow_list",
                    "allowed_tool_names": [
                        "search_contacts",
                        "send_message_with_phone_number"
                    ]
                }
            }
        },
        "router": {
            "type": "llm",
            "model_name": "gpt-4o-mini",
            "agent_descriptions": {
                "planner": "Plans which tools to use and in what order",
                "executor": "Executes tool calls for messaging and contacts"
            }
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tool_sandbox.roles.agent_framework_agent import AgentFrameworkAgent
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.multi_agent import (
    AgentRouter,
    KeywordRouter,
    LLMRouter,
    MultiAgentRole,
    RoundRobinRouter,
    ToolBasedRouter,
)
from tool_sandbox.roles.tool_filter import (
    AllowListToolFilter,
    CompositeToolFilter,
    DenyListToolFilter,
    KeywordToolFilter,
    LLMClassifierToolFilter,
    MLClassifierToolFilter,
    ToolFilter,
)
from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent
from tool_sandbox.roles.tool_serializer import (
    CompactDescriptionSerializer,
    CompositeToolSerializer,
    DescriptionPrefixSerializer,
    IdentityToolSerializer,
    JSONSchemaAnnotationSerializer,
    MinimalSchemaSerializer,
    ToolSerializer,
    XMLToolSerializer,
)
from tool_sandbox.roles.tool_serialized_agent import ToolSerializedAgent


def _build_tool_filter(config: dict[str, Any]) -> ToolFilter:
    """Construct a :class:`ToolFilter` from a JSON config dict.

    Args:
        config: Dictionary with a ``"type"`` key and type-specific parameters.

    Returns:
        A :class:`ToolFilter` instance.

    Raises:
        ValueError: If the type is unknown.
    """
    filter_type = config["type"]

    if filter_type == "allow_list":
        return AllowListToolFilter(
            allowed_tool_names=config["allowed_tool_names"],
        )

    if filter_type == "deny_list":
        return DenyListToolFilter(
            denied_tool_names=config["denied_tool_names"],
        )

    if filter_type == "keyword":
        return KeywordToolFilter(
            keyword_to_tools=config["keyword_to_tools"],
            include_unmatched=config.get("include_unmatched", False),
            default_tools=config.get("default_tools"),
        )

    if filter_type == "llm_classifier":
        return LLMClassifierToolFilter(
            model_name=config.get("model_name", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            max_tools=config.get("max_tools"),
            always_include=config.get("always_include"),
        )

    if filter_type == "ml_classifier":
        return MLClassifierToolFilter(
            endpoint_url=config.get("endpoint_url", "http://localhost:5050/predict"),
            threshold=config.get("threshold", 0.5),
            always_include=config.get("always_include"),
            timeout=config.get("timeout", 10.0),
        )

    if filter_type == "composite":
        sub_filters = [_build_tool_filter(fc) for fc in config["filters"]]
        return CompositeToolFilter(filters=sub_filters)

    raise ValueError(f"Unknown tool filter type: '{filter_type}'")


def _build_tool_serializer(config: dict[str, Any]) -> ToolSerializer:
    """Construct a :class:`ToolSerializer` from a JSON config dict.

    Args:
        config: Dictionary with a ``"type"`` key and type-specific parameters.

    Returns:
        A :class:`ToolSerializer` instance.

    Raises:
        ValueError: If the type is unknown.
    """
    ser_type = config["type"]

    if ser_type == "identity":
        return IdentityToolSerializer()

    if ser_type == "compact_description":
        return CompactDescriptionSerializer(
            max_tool_desc_length=config.get("max_tool_desc_length", 80),
            max_param_desc_length=config.get("max_param_desc_length", 60),
            ellipsis_marker=config.get("ellipsis_marker", "..."),
        )

    if ser_type == "minimal_schema":
        return MinimalSchemaSerializer(
            keep_tool_description=config.get("keep_tool_description", True),
            keep_param_types=config.get("keep_param_types", True),
        )

    if ser_type == "json_schema_annotation":
        return JSONSchemaAnnotationSerializer(
            annotations=config["annotations"],
        )

    if ser_type == "description_prefix":
        return DescriptionPrefixSerializer(
            prefix=config["prefix"],
        )

    if ser_type == "xml":
        return XMLToolSerializer()

    if ser_type == "composite":
        sub_serializers = [_build_tool_serializer(sc) for sc in config["serializers"]]
        return CompositeToolSerializer(serializers=sub_serializers)

    raise ValueError(f"Unknown tool serializer type: '{ser_type}'")


def _build_router(config: dict[str, Any]) -> AgentRouter:
    """Construct an :class:`AgentRouter` from a JSON config dict.

    Args:
        config: Dictionary with a ``"type"`` key and type-specific parameters.

    Returns:
        An :class:`AgentRouter` instance.

    Raises:
        ValueError: If the type is unknown.
    """
    router_type = config["type"]

    if router_type == "round_robin":
        return RoundRobinRouter(order=config.get("order"))

    if router_type == "keyword":
        return KeywordRouter(
            keyword_to_agent=config["keyword_to_agent"],
            default_agent=config["default_agent"],
        )

    if router_type == "llm":
        return LLMRouter(
            model_name=config.get("model_name", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            agent_descriptions=config.get("agent_descriptions"),
        )

    if router_type == "tool_based":
        return ToolBasedRouter(default_agent=config.get("default_agent"))

    raise ValueError(f"Unknown router type: '{router_type}'")


def _build_simple_agent(agent_type_str: str) -> BaseRole:
    """Create a plain agent from its ``RoleImplType`` string name.

    This import is deferred to avoid circular imports with ``cli/utils.py``.
    """
    from tool_sandbox.cli.utils import AGENT_TYPE_TO_FACTORY, RoleImplType

    role_impl = RoleImplType(agent_type_str)
    return AGENT_TYPE_TO_FACTORY[role_impl]()


def build_agent_from_config(config: dict[str, Any]) -> BaseRole:
    """Recursively build an agent (possibly nested) from a JSON config dict.

    Args:
        config: Agent configuration dictionary.  Must contain a ``"type"`` key.

    Returns:
        A :class:`BaseRole` instance ready to be used in a scenario.

    Raises:
        ValueError: If the type is unknown.
    """
    agent_type = config["type"]

    # --- Tool-filtered agent ---
    if agent_type == "tool_filtered":
        inner_config = config["inner_agent"]
        if isinstance(inner_config, str):
            inner_agent = _build_simple_agent(inner_config)
        else:
            inner_agent = build_agent_from_config(inner_config)
        tool_filter = _build_tool_filter(config["tool_filter"])
        return ToolFilteredAgent(inner_agent=inner_agent, tool_filter=tool_filter)

    # --- Tool-serialized agent ---
    if agent_type == "tool_serialized":
        inner_config = config["inner_agent"]
        if isinstance(inner_config, str):
            inner_agent = _build_simple_agent(inner_config)
        else:
            inner_agent = build_agent_from_config(inner_config)
        tool_serializer = _build_tool_serializer(config["tool_serializer"])
        return ToolSerializedAgent(
            inner_agent=inner_agent, tool_serializer=tool_serializer
        )

    # --- Multi-agent ---
    if agent_type == "multi_agent":
        agents: dict[str, BaseRole] = {}
        for name, agent_cfg in config["agents"].items():
            if isinstance(agent_cfg, str):
                agents[name] = _build_simple_agent(agent_cfg)
            else:
                agents[name] = build_agent_from_config(agent_cfg)
        router = _build_router(config["router"])
        return MultiAgentRole(agents=agents, router=router)

    if agent_type == "agent_framework":
        inner_config = config["inner_agent"]
        if isinstance(inner_config, str):
            inner_agent = _build_simple_agent(inner_config)
        else:
            inner_agent = build_agent_from_config(inner_config)
        return AgentFrameworkAgent(inner_agent=inner_agent)

    # --- Plain agent (by RoleImplType name) ---
    return _build_simple_agent(agent_type)


def load_agent_config(path: Path) -> BaseRole:
    """Load an agent configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        The constructed agent.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return build_agent_from_config(config)
