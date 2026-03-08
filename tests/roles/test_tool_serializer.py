"""Tests for tool serialization middleware.

Covers:
- Every concrete ToolSerializer implementation (serialization correctness).
- Roundtrip: serialize with each serializer → verify output structure is valid
  OpenAI tool schema so models can still call tools.
- CompositeToolSerializer chaining.
- ToolSerializedAgent wrapper: delegation, model_name, get_available_tools,
  serialize_tools.
- BaseRole.serialize_tools with and without a serializer attached.
- agent_config integration: _build_tool_serializer and build_agent_from_config
  with ``"tool_serialized"`` type.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.tool_conversion import (
    convert_to_openai_tool,
    convert_to_openai_tools,
)
from tool_sandbox.roles.base_role import BaseRole
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _execution_context() -> None:
    """Set up a fresh execution context for each test."""
    ctx = ExecutionContext()
    set_current_context(ctx)


def _make_dummy_tool(
    name: str,
    doc: str = "A dummy tool that does nothing.",
) -> Callable[..., Any]:
    """Create a trivially callable tool with a name and docstring."""

    def fn() -> None:
        pass

    fn.__name__ = name
    fn.__doc__ = doc
    fn.is_tool = True  # type: ignore[attr-defined]
    fn.visible_to = (RoleType.AGENT,)  # type: ignore[attr-defined]
    return fn


def _make_typed_tool() -> Callable[..., Any]:
    """Create a tool with typed parameters and a Google-style docstring."""

    def send_message(recipient: str, body: str, urgent: bool = False) -> str:
        """Send a text message to someone.

        Args:
            recipient: The name of the person to message.
            body:      The message content.
            urgent:    Whether the message is urgent.

        Returns:
            Confirmation string.
        """
        return "sent"

    send_message.is_tool = True  # type: ignore[attr-defined]
    send_message.visible_to = (RoleType.AGENT,)  # type: ignore[attr-defined]
    return send_message


def _make_tools() -> dict[str, Callable[..., Any]]:
    return {
        "search_contacts": _make_dummy_tool(
            "search_contacts", "Search for contacts in the address book."
        ),
        "send_message": _make_typed_tool(),
        "set_wifi": _make_dummy_tool(
            "set_wifi", "Toggle the WiFi setting on or off."
        ),
    }


def _sample_openai_tools() -> list[dict[str, Any]]:
    """Pre-built OpenAI-format tool dicts for unit tests that don't need
    the full convert_to_openai_tools pipeline."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_contacts",
                "description": "Search for contacts in the address book.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Name or phone number to search for.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a text message to someone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "The name of the person to message.",
                        },
                        "body": {
                            "type": "string",
                            "description": "The message content.",
                        },
                        "urgent": {
                            "type": "boolean",
                            "description": "Whether the message is urgent.",
                        },
                    },
                    "required": ["recipient", "body"],
                },
            },
        },
    ]


def _make_messages(content: str = "Send a message to John") -> list[Message]:
    return [
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content=content,
        ),
    ]


def _assert_valid_openai_tools(tools: list[dict[str, Any]]) -> None:
    """Assert every tool dict conforms to the minimal OpenAI tool-calling schema."""
    for tool in tools:
        assert "type" in tool, "tool missing 'type'"
        assert tool["type"] == "function"
        func = tool["function"]
        assert "name" in func, "function missing 'name'"
        # parameters and description may be absent in some serializers,
        # but if present they must be dicts / strings.
        if "parameters" in func:
            assert isinstance(func["parameters"], dict)
        if "description" in func:
            assert isinstance(func["description"], str)


# ===================================================================
# IdentityToolSerializer
# ===================================================================


class TestIdentityToolSerializer:
    def test_returns_tools_unchanged(self) -> None:
        tools = _sample_openai_tools()
        ser = IdentityToolSerializer()
        result = ser.serialize_tools(tools)
        assert result == tools

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = IdentityToolSerializer()
        _assert_valid_openai_tools(ser.serialize_tools(tools))

    def test_empty_list(self) -> None:
        ser = IdentityToolSerializer()
        assert ser.serialize_tools([]) == []


# ===================================================================
# CompactDescriptionSerializer
# ===================================================================


class TestCompactDescriptionSerializer:
    def test_truncates_tool_description(self) -> None:
        tools = _sample_openai_tools()
        ser = CompactDescriptionSerializer(max_tool_desc_length=10)
        result = ser.serialize_tools(tools)
        for tool in result:
            desc = tool["function"].get("description", "")
            assert len(desc) <= 10 + len("...")  # original may be shorter

    def test_truncates_param_descriptions(self) -> None:
        tools = _sample_openai_tools()
        ser = CompactDescriptionSerializer(max_param_desc_length=15)
        result = ser.serialize_tools(tools)
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pname, pspec in props.items():
                desc = pspec.get("description", "")
                assert len(desc) <= 15 + len("...")

    def test_no_limit(self) -> None:
        tools = _sample_openai_tools()
        original_descs = [t["function"]["description"] for t in tools]
        ser = CompactDescriptionSerializer(
            max_tool_desc_length=None, max_param_desc_length=None
        )
        result = ser.serialize_tools(tools)
        for tool, orig_desc in zip(result, original_descs):
            assert tool["function"]["description"] == orig_desc

    def test_does_not_mutate_original(self) -> None:
        tools = _sample_openai_tools()
        import copy

        original = copy.deepcopy(tools)
        ser = CompactDescriptionSerializer(max_tool_desc_length=5)
        ser.serialize_tools(tools)
        assert tools == original

    def test_custom_ellipsis(self) -> None:
        tools = _sample_openai_tools()
        ser = CompactDescriptionSerializer(
            max_tool_desc_length=10, ellipsis_marker="~"
        )
        result = ser.serialize_tools(tools)
        for tool in result:
            desc = tool["function"]["description"]
            if len(desc) > 10:
                assert desc.endswith("~")

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = CompactDescriptionSerializer(max_tool_desc_length=5, max_param_desc_length=5)
        _assert_valid_openai_tools(ser.serialize_tools(tools))


# ===================================================================
# MinimalSchemaSerializer
# ===================================================================


class TestMinimalSchemaSerializer:
    def test_removes_param_descriptions(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer()
        result = ser.serialize_tools(tools)
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "description" not in pspec

    def test_keeps_tool_description_by_default(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer()
        result = ser.serialize_tools(tools)
        for tool in result:
            assert "description" in tool["function"]

    def test_removes_tool_description(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer(keep_tool_description=False)
        result = ser.serialize_tools(tools)
        for tool in result:
            assert "description" not in tool["function"]

    def test_removes_param_types(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer(keep_param_types=False)
        result = ser.serialize_tools(tools)
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "type" not in pspec

    def test_keeps_param_types_by_default(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer()
        result = ser.serialize_tools(tools)
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "type" in pspec

    def test_preserves_required(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer()
        result = ser.serialize_tools(tools)
        # send_message has required: ["recipient", "body"]
        send_msg = [t for t in result if t["function"]["name"] == "send_message"][0]
        assert set(send_msg["function"]["parameters"]["required"]) == {
            "recipient",
            "body",
        }

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer(keep_tool_description=False, keep_param_types=False)
        _assert_valid_openai_tools(ser.serialize_tools(tools))


# ===================================================================
# JSONSchemaAnnotationSerializer
# ===================================================================


class TestJSONSchemaAnnotationSerializer:
    def test_adds_annotations(self) -> None:
        tools = _sample_openai_tools()
        ser = JSONSchemaAnnotationSerializer(
            annotations={
                "send_message": {
                    "recipient": {"examples": ["Alice", "Bob"]},
                    "body": {"default": "Hello!"},
                }
            }
        )
        result = ser.serialize_tools(tools)
        send_msg = [t for t in result if t["function"]["name"] == "send_message"][0]
        props = send_msg["function"]["parameters"]["properties"]
        assert props["recipient"]["examples"] == ["Alice", "Bob"]
        assert props["body"]["default"] == "Hello!"
        # Original fields preserved
        assert props["recipient"]["type"] == "string"

    def test_ignores_unknown_tools(self) -> None:
        tools = _sample_openai_tools()
        ser = JSONSchemaAnnotationSerializer(
            annotations={"nonexistent_tool": {"x": {"y": 1}}}
        )
        result = ser.serialize_tools(tools)
        assert result == _sample_openai_tools()

    def test_ignores_unknown_params(self) -> None:
        tools = _sample_openai_tools()
        ser = JSONSchemaAnnotationSerializer(
            annotations={"send_message": {"nonexistent_param": {"foo": "bar"}}}
        )
        result = ser.serialize_tools(tools)
        send_msg = [t for t in result if t["function"]["name"] == "send_message"][0]
        assert "nonexistent_param" not in send_msg["function"]["parameters"]["properties"]

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = JSONSchemaAnnotationSerializer(
            annotations={"send_message": {"body": {"minLength": 1}}}
        )
        _assert_valid_openai_tools(ser.serialize_tools(tools))


# ===================================================================
# DescriptionPrefixSerializer
# ===================================================================


class TestDescriptionPrefixSerializer:
    def test_prepends_prefix(self) -> None:
        tools = _sample_openai_tools()
        prefix = "IMPORTANT:"
        ser = DescriptionPrefixSerializer(prefix=prefix)
        result = ser.serialize_tools(tools)
        for tool in result:
            assert tool["function"]["description"].startswith(prefix)

    def test_prefix_with_empty_description(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "empty_desc",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        ser = DescriptionPrefixSerializer(prefix="NOTE:")
        result = ser.serialize_tools(tools)
        assert result[0]["function"]["description"] == "NOTE:"

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = DescriptionPrefixSerializer(prefix="INFO:")
        _assert_valid_openai_tools(ser.serialize_tools(tools))


# ===================================================================
# XMLToolSerializer
# ===================================================================


class TestXMLToolSerializer:
    def test_embeds_xml_in_description(self) -> None:
        tools = _sample_openai_tools()
        ser = XMLToolSerializer()
        result = ser.serialize_tools(tools)
        for tool in result:
            desc = tool["function"]["description"]
            name = tool["function"]["name"]
            assert f'<tool name="{name}">' in desc
            assert "</tool>" in desc
            assert "<parameters>" in desc

    def test_param_attributes(self) -> None:
        tools = _sample_openai_tools()
        ser = XMLToolSerializer()
        result = ser.serialize_tools(tools)
        send_msg = [t for t in result if t["function"]["name"] == "send_message"][0]
        desc = send_msg["function"]["description"]
        assert 'name="recipient"' in desc
        assert 'type="string"' in desc
        assert 'required="true"' in desc

    def test_preserves_original_parameters(self) -> None:
        """The original parameters dict must still be present (for API compatibility)."""
        tools = _sample_openai_tools()
        ser = XMLToolSerializer()
        result = ser.serialize_tools(tools)
        for tool in result:
            assert "parameters" in tool["function"]

    def test_output_is_valid_openai_schema(self) -> None:
        tools = _sample_openai_tools()
        ser = XMLToolSerializer()
        _assert_valid_openai_tools(ser.serialize_tools(tools))


# ===================================================================
# CompositeToolSerializer
# ===================================================================


class TestCompositeToolSerializer:
    def test_chains_serializers_in_order(self) -> None:
        """compact → prefix: description gets truncated then prefixed."""
        tools = _sample_openai_tools()
        compact = CompactDescriptionSerializer(max_tool_desc_length=10)
        prefix = DescriptionPrefixSerializer(prefix="NOTE:")
        composite = CompositeToolSerializer(serializers=[compact, prefix])
        result = composite.serialize_tools(tools)
        for tool in result:
            desc = tool["function"]["description"]
            assert desc.startswith("NOTE:")

    def test_empty_composite(self) -> None:
        tools = _sample_openai_tools()
        composite = CompositeToolSerializer(serializers=[])
        assert composite.serialize_tools(tools) == tools

    def test_single_serializer(self) -> None:
        tools = _sample_openai_tools()
        identity = IdentityToolSerializer()
        composite = CompositeToolSerializer(serializers=[identity])
        assert composite.serialize_tools(tools) == tools

    def test_triple_chain(self) -> None:
        tools = _sample_openai_tools()
        s1 = MinimalSchemaSerializer()  # strip param descriptions
        s2 = CompactDescriptionSerializer(max_tool_desc_length=20)  # truncate tool desc
        s3 = DescriptionPrefixSerializer(prefix="!")  # prefix
        composite = CompositeToolSerializer(serializers=[s1, s2, s3])
        result = composite.serialize_tools(tools)
        for tool in result:
            desc = tool["function"]["description"]
            assert desc.startswith("!")
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "description" not in pspec


# ===================================================================
# Roundtrip: convert_to_openai_tools -> serialize -> valid schema
# ===================================================================


class TestSerializationRoundtrip:
    """Ensure that converting real callables to OpenAI tools and then
    applying a serializer still yields structurally valid tool dicts."""

    @pytest.fixture()
    def openai_tools_from_callables(self) -> list[dict[str, Any]]:
        return convert_to_openai_tools(_make_tools())

    @pytest.mark.parametrize(
        "serializer",
        [
            IdentityToolSerializer(),
            CompactDescriptionSerializer(max_tool_desc_length=20, max_param_desc_length=10),
            MinimalSchemaSerializer(),
            MinimalSchemaSerializer(keep_tool_description=False, keep_param_types=False),
            DescriptionPrefixSerializer(prefix="TEST:"),
            XMLToolSerializer(),
            CompositeToolSerializer(
                serializers=[
                    MinimalSchemaSerializer(),
                    CompactDescriptionSerializer(max_tool_desc_length=30),
                ]
            ),
        ],
        ids=[
            "identity",
            "compact",
            "minimal",
            "minimal_no_desc_no_types",
            "prefix",
            "xml",
            "composite",
        ],
    )
    def test_roundtrip_produces_valid_schema(
        self,
        serializer: ToolSerializer,
        openai_tools_from_callables: list[dict[str, Any]],
    ) -> None:
        result = serializer.serialize_tools(openai_tools_from_callables)
        _assert_valid_openai_tools(result)
        # Tool names are preserved (critical for deserialization)
        original_names = {
            t["function"]["name"] for t in openai_tools_from_callables
        }
        result_names = {t["function"]["name"] for t in result}
        assert result_names == original_names

    def test_roundtrip_preserves_tool_count(
        self,
        openai_tools_from_callables: list[dict[str, Any]],
    ) -> None:
        ser = CompactDescriptionSerializer(max_tool_desc_length=5)
        result = ser.serialize_tools(openai_tools_from_callables)
        assert len(result) == len(openai_tools_from_callables)


# ===================================================================
# BaseRole.serialize_tools (without and with serializer)
# ===================================================================


class TestBaseRoleSerializeTools:
    def test_without_serializer(self) -> None:
        """With no serializer, serialize_tools should produce the same as
        convert_to_openai_tools."""
        role = BaseRole()
        role.role_type = RoleType.AGENT
        tools = _make_tools()
        result = role.serialize_tools(tools)
        expected = convert_to_openai_tools(tools)
        assert result == expected

    def test_with_serializer(self) -> None:
        """With a serializer attached, the output should be transformed."""
        role = BaseRole()
        role.role_type = RoleType.AGENT
        role._tool_serializer = MinimalSchemaSerializer()
        tools = _make_tools()
        result = role.serialize_tools(tools)
        # Param descriptions should have been stripped
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "description" not in pspec

    def test_identity_serializer_matches_no_serializer(self) -> None:
        role_plain = BaseRole()
        role_plain.role_type = RoleType.AGENT
        role_ser = BaseRole()
        role_ser.role_type = RoleType.AGENT
        role_ser._tool_serializer = IdentityToolSerializer()
        tools = _make_tools()
        assert role_plain.serialize_tools(tools) == role_ser.serialize_tools(tools)


# ===================================================================
# ToolSerializedAgent wrapper
# ===================================================================


class _DummyAgent(BaseRole):
    """Minimal agent stub for testing the wrapper."""

    role_type: RoleType = RoleType.AGENT
    model_name: str = "dummy-model"

    def __init__(self) -> None:
        self._respond_called = False
        self._reset_called = False
        self._teardown_called = False

    def respond(self, ending_index: Optional[int] = None) -> None:
        self._respond_called = True

    def reset(self) -> None:
        self._reset_called = True

    def teardown(self) -> None:
        self._teardown_called = True


class TestToolSerializedAgent:
    def test_delegates_respond(self) -> None:
        inner = _DummyAgent()
        agent = ToolSerializedAgent(inner, IdentityToolSerializer())
        agent.respond()
        assert inner._respond_called

    def test_delegates_reset(self) -> None:
        inner = _DummyAgent()
        agent = ToolSerializedAgent(inner, IdentityToolSerializer())
        agent.reset()
        assert inner._reset_called

    def test_delegates_teardown(self) -> None:
        inner = _DummyAgent()
        agent = ToolSerializedAgent(inner, IdentityToolSerializer())
        agent.teardown()
        assert inner._teardown_called

    def test_model_name_includes_serializer(self) -> None:
        inner = _DummyAgent()
        ser = CompactDescriptionSerializer()
        agent = ToolSerializedAgent(inner, ser)
        assert "dummy-model" in agent.model_name
        assert "CompactDescriptionSerializer" in agent.model_name

    def test_serializer_attached_to_inner_agent(self) -> None:
        inner = _DummyAgent()
        ser = MinimalSchemaSerializer()
        ToolSerializedAgent(inner, ser)
        assert inner._tool_serializer is ser

    def test_serialize_tools_delegates(self) -> None:
        inner = _DummyAgent()
        ser = MinimalSchemaSerializer()
        agent = ToolSerializedAgent(inner, ser)
        tools = _make_tools()
        result = agent.serialize_tools(tools)
        # Should have param descriptions stripped
        for tool in result:
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "description" not in pspec

    def test_composability_with_tool_filter(self) -> None:
        """A ToolSerializedAgent can wrap a ToolFilteredAgent (or vice versa)."""
        from tool_sandbox.roles.tool_filter import AllowListToolFilter
        from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent

        inner = _DummyAgent()
        filt = AllowListToolFilter(["send_message"])
        filtered = ToolFilteredAgent(inner_agent=inner, tool_filter=filt)
        ser = MinimalSchemaSerializer()
        agent = ToolSerializedAgent(inner_agent=filtered, tool_serializer=ser)
        # Wrapper should delegate
        agent.respond()
        assert inner._respond_called
        # model_name should chain
        assert "filtered" in filtered.model_name
        assert "serialized" in agent.model_name


# ===================================================================
# agent_config integration
# ===================================================================


class TestAgentConfigSerialization:
    """Tests for _build_tool_serializer and build_agent_from_config."""

    def test_build_identity(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer({"type": "identity"})
        assert isinstance(ser, IdentityToolSerializer)

    def test_build_compact_description(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer(
            {
                "type": "compact_description",
                "max_tool_desc_length": 50,
                "max_param_desc_length": 30,
            }
        )
        assert isinstance(ser, CompactDescriptionSerializer)
        assert ser.max_tool_desc_length == 50
        assert ser.max_param_desc_length == 30

    def test_build_minimal_schema(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer(
            {"type": "minimal_schema", "keep_tool_description": False}
        )
        assert isinstance(ser, MinimalSchemaSerializer)
        assert ser.keep_tool_description is False

    def test_build_json_schema_annotation(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer(
            {
                "type": "json_schema_annotation",
                "annotations": {"my_tool": {"param1": {"examples": [1, 2]}}},
            }
        )
        assert isinstance(ser, JSONSchemaAnnotationSerializer)

    def test_build_description_prefix(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer({"type": "description_prefix", "prefix": "!"})
        assert isinstance(ser, DescriptionPrefixSerializer)
        assert ser.prefix == "!"

    def test_build_xml(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer({"type": "xml"})
        assert isinstance(ser, XMLToolSerializer)

    def test_build_composite(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        ser = _build_tool_serializer(
            {
                "type": "composite",
                "serializers": [
                    {"type": "minimal_schema"},
                    {"type": "compact_description", "max_tool_desc_length": 20},
                ],
            }
        )
        assert isinstance(ser, CompositeToolSerializer)
        assert len(ser.serializers) == 2

    def test_build_unknown_raises(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_serializer

        with pytest.raises(ValueError, match="Unknown tool serializer type"):
            _build_tool_serializer({"type": "nonexistent"})

    def test_build_agent_from_config_tool_serialized(self) -> None:
        """build_agent_from_config should handle 'tool_serialized' type."""
        from tool_sandbox.cli.agent_config import build_agent_from_config

        # We mock the simple agent builder to avoid importing real model deps.
        with patch(
            "tool_sandbox.cli.agent_config._build_simple_agent"
        ) as mock_build:
            mock_build.return_value = _DummyAgent()
            agent = build_agent_from_config(
                {
                    "type": "tool_serialized",
                    "inner_agent": "SomeFakeAgent",
                    "tool_serializer": {"type": "minimal_schema"},
                }
            )
        assert isinstance(agent, ToolSerializedAgent)
        assert isinstance(agent.tool_serializer, MinimalSchemaSerializer)

    def test_build_agent_from_config_nested_filter_and_serializer(self) -> None:
        """A tool_serialized agent can wrap a tool_filtered agent via config."""
        from tool_sandbox.cli.agent_config import build_agent_from_config
        from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent

        with patch(
            "tool_sandbox.cli.agent_config._build_simple_agent"
        ) as mock_build:
            mock_build.return_value = _DummyAgent()
            agent = build_agent_from_config(
                {
                    "type": "tool_serialized",
                    "inner_agent": {
                        "type": "tool_filtered",
                        "inner_agent": "SomeFakeAgent",
                        "tool_filter": {
                            "type": "allow_list",
                            "allowed_tool_names": ["send_message"],
                        },
                    },
                    "tool_serializer": {"type": "compact_description"},
                }
            )
        assert isinstance(agent, ToolSerializedAgent)
        assert isinstance(agent.inner_agent, ToolFilteredAgent)


# ===================================================================
# Deserialization: serialized tools can still be called
# ===================================================================


class TestDeserializationStillWorks:
    """After serializing tools, the tool names and parameter structure must
    remain intact enough that openai_tool_call_to_python_code can convert a
    model response back to executable Python."""

    def test_tool_names_preserved_after_serialization(self) -> None:
        """Tool names must be unchanged by any serializer."""
        tools = _sample_openai_tools()
        original_names = {t["function"]["name"] for t in tools}

        serializers: list[ToolSerializer] = [
            IdentityToolSerializer(),
            CompactDescriptionSerializer(max_tool_desc_length=5),
            MinimalSchemaSerializer(keep_tool_description=False),
            DescriptionPrefixSerializer(prefix="X:"),
            XMLToolSerializer(),
        ]
        for ser in serializers:
            result_names = {t["function"]["name"] for t in ser.serialize_tools(tools)}
            assert result_names == original_names, (
                f"{type(ser).__name__} changed tool names"
            )

    def test_parameter_names_preserved(self) -> None:
        """Parameter names in 'properties' must persist so the model can
        reference them in its tool call."""
        tools = _sample_openai_tools()
        send_msg = [t for t in tools if t["function"]["name"] == "send_message"][0]
        original_params = set(
            send_msg["function"]["parameters"]["properties"].keys()
        )

        serializers: list[ToolSerializer] = [
            CompactDescriptionSerializer(max_tool_desc_length=5, max_param_desc_length=5),
            MinimalSchemaSerializer(keep_param_types=False),
            JSONSchemaAnnotationSerializer(
                annotations={"send_message": {"body": {"examples": ["hi"]}}}
            ),
            DescriptionPrefixSerializer(prefix="X:"),
            XMLToolSerializer(),
        ]
        for ser in serializers:
            result = ser.serialize_tools(tools)
            send_msg_r = [
                t for t in result if t["function"]["name"] == "send_message"
            ][0]
            result_params = set(
                send_msg_r["function"]["parameters"]["properties"].keys()
            )
            assert result_params == original_params, (
                f"{type(ser).__name__} changed param names"
            )

    def test_required_fields_preserved(self) -> None:
        """The 'required' array must survive serialization."""
        tools = _sample_openai_tools()
        send_msg = [t for t in tools if t["function"]["name"] == "send_message"][0]
        original_required = set(send_msg["function"]["parameters"]["required"])

        serializers: list[ToolSerializer] = [
            CompactDescriptionSerializer(),
            MinimalSchemaSerializer(),
            DescriptionPrefixSerializer(prefix="X:"),
            XMLToolSerializer(),
        ]
        for ser in serializers:
            result = ser.serialize_tools(tools)
            send_msg_r = [
                t for t in result if t["function"]["name"] == "send_message"
            ][0]
            result_required = set(send_msg_r["function"]["parameters"]["required"])
            assert result_required == original_required, (
                f"{type(ser).__name__} changed required fields"
            )

    def test_openai_tool_call_to_python_code_after_serialization(self) -> None:
        """Simulate: tools serialized → model returns tool call → deserialize
        with openai_tool_call_to_python_code → valid Python code."""
        from unittest.mock import MagicMock

        from tool_sandbox.common.message_conversion import (
            openai_tool_call_to_python_code,
        )

        # Build a fake ChatCompletionMessageToolCall
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "send_message"
        tool_call.function.arguments = json.dumps(
            {"recipient": "Alice", "body": "Hello!"}
        )

        # Serialize tools with a non-trivial serializer
        tools = _sample_openai_tools()
        ser = CompactDescriptionSerializer(max_tool_desc_length=5, max_param_desc_length=5)
        serialized = ser.serialize_tools(tools)

        # The available tool names come from the serialized output
        available = {t["function"]["name"] for t in serialized}

        # Deserialization should still work
        code = openai_tool_call_to_python_code(
            tool_call,
            available_tool_names=available,
            execution_facing_tool_name="send_message",
        )
        assert "send_message" in code
        assert "Alice" in code
        assert "Hello!" in code

    def test_deserialization_with_xml_serializer(self) -> None:
        """XML serializer changes description but tool names / params stay
        the same, so deserialization should work."""
        from unittest.mock import MagicMock

        from tool_sandbox.common.message_conversion import (
            openai_tool_call_to_python_code,
        )

        tool_call = MagicMock()
        tool_call.id = "call_456"
        tool_call.function.name = "search_contacts"
        tool_call.function.arguments = json.dumps({"query": "John"})

        tools = _sample_openai_tools()
        ser = XMLToolSerializer()
        serialized = ser.serialize_tools(tools)

        available = {t["function"]["name"] for t in serialized}
        code = openai_tool_call_to_python_code(
            tool_call,
            available_tool_names=available,
            execution_facing_tool_name="search_contacts",
        )
        assert "search_contacts" in code
        assert "John" in code

    def test_deserialization_with_minimal_schema(self) -> None:
        """Minimal schema strips descriptions but the model should still
        be able to call tools correctly."""
        from unittest.mock import MagicMock

        from tool_sandbox.common.message_conversion import (
            openai_tool_call_to_python_code,
        )

        tool_call = MagicMock()
        tool_call.id = "call_789"
        tool_call.function.name = "send_message"
        tool_call.function.arguments = json.dumps(
            {"recipient": "Bob", "body": "Hi", "urgent": True}
        )

        tools = _sample_openai_tools()
        ser = MinimalSchemaSerializer(keep_tool_description=False, keep_param_types=False)
        serialized = ser.serialize_tools(tools)

        available = {t["function"]["name"] for t in serialized}
        code = openai_tool_call_to_python_code(
            tool_call,
            available_tool_names=available,
            execution_facing_tool_name="send_message",
        )
        assert "send_message" in code
        assert "Bob" in code
        assert "True" in code


# ===================================================================
# End-to-end: convert real callables → serialize → validate → deserialize
# ===================================================================


class TestEndToEndWithRealCallables:
    """Full pipeline with actual Python callables rather than hand-built dicts."""

    def test_identity_pipeline(self) -> None:
        tools = _make_tools()
        role = BaseRole()
        role.role_type = RoleType.AGENT
        role._tool_serializer = IdentityToolSerializer()
        result = role.serialize_tools(tools)
        expected = convert_to_openai_tools(tools)
        assert result == expected
        _assert_valid_openai_tools(result)

    def test_compact_pipeline(self) -> None:
        tools = _make_tools()
        role = BaseRole()
        role.role_type = RoleType.AGENT
        role._tool_serializer = CompactDescriptionSerializer(
            max_tool_desc_length=15, max_param_desc_length=10
        )
        result = role.serialize_tools(tools)
        _assert_valid_openai_tools(result)
        # Check truncation happened
        for tool in result:
            desc = tool["function"].get("description", "")
            # 15 + len("...") = 18 max
            assert len(desc) <= 18

    def test_xml_pipeline(self) -> None:
        tools = _make_tools()
        role = BaseRole()
        role.role_type = RoleType.AGENT
        role._tool_serializer = XMLToolSerializer()
        result = role.serialize_tools(tools)
        _assert_valid_openai_tools(result)
        for tool in result:
            assert "<tool" in tool["function"]["description"]

    def test_composite_pipeline(self) -> None:
        tools = _make_tools()
        role = BaseRole()
        role.role_type = RoleType.AGENT
        role._tool_serializer = CompositeToolSerializer(
            serializers=[
                MinimalSchemaSerializer(),
                DescriptionPrefixSerializer(prefix="[TOOL]"),
            ]
        )
        result = role.serialize_tools(tools)
        _assert_valid_openai_tools(result)
        for tool in result:
            assert tool["function"]["description"].startswith("[TOOL]")
            props = tool["function"].get("parameters", {}).get("properties", {})
            for pspec in props.values():
                assert "description" not in pspec
