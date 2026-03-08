"""Tool serialization middleware for customising how tools are presented to an LLM.

A :class:`ToolSerializer` sits between the standard
:func:`~tool_sandbox.common.tool_conversion.convert_to_openai_tools` output and
the LLM API call.  When attached to a
:class:`~tool_sandbox.roles.base_role.BaseRole` instance (via
``_tool_serializer``), :meth:`BaseRole.serialize_tools` will apply the
serializer's :meth:`serialize_tools` transformation after the default
LangChain-derived conversion.

This makes it possible to experiment with different tool-description formats
(compact schemas, XML representations, etc.) without altering the core
conversion logic.
"""

from __future__ import annotations

import copy
import json
import textwrap
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Optional, Sequence

LOGGER = getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ToolSerializer(ABC):
    """Base class for all tool serializers.

    Subclasses must implement :meth:`serialize_tools` which receives the list
    of OpenAI-format tool dicts (as produced by
    :func:`~tool_sandbox.common.tool_conversion.convert_to_openai_tools`) and
    returns a (possibly modified) list of tool dicts.
    """

    @abstractmethod
    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Transform the OpenAI-format tool definitions before they are sent to
        the model.

        Args:
            tools: List of tool dicts, each with ``{"type": "function",
                   "function": {"name": ..., "description": ...,
                   "parameters": ...}}``.

        Returns:
            A (possibly modified) list of tool dicts in the same top-level
            format.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class IdentityToolSerializer(ToolSerializer):
    """Pass-through serializer that returns tools unchanged.

    Useful as a baseline for comparison or as a no-op placeholder.
    """

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return tools


class CompactDescriptionSerializer(ToolSerializer):
    """Truncate tool and parameter descriptions to a maximum character length.

    This lets you test whether an LLM can still call tools correctly with
    shorter / degraded descriptions.

    Args:
        max_tool_desc_length:  Maximum characters for the top-level function
                               description.  ``None`` means no limit.
        max_param_desc_length: Maximum characters for each parameter's
                               description.  ``None`` means no limit.
        ellipsis_marker:       String appended when a description is truncated.
    """

    def __init__(
        self,
        max_tool_desc_length: Optional[int] = 80,
        max_param_desc_length: Optional[int] = 60,
        ellipsis_marker: str = "...",
    ) -> None:
        self.max_tool_desc_length = max_tool_desc_length
        self.max_param_desc_length = max_param_desc_length
        self.ellipsis_marker = ellipsis_marker

    def _truncate(self, text: str, max_len: Optional[int]) -> str:
        if max_len is None or len(text) <= max_len:
            return text
        cut = max(0, max_len - len(self.ellipsis_marker))
        return text[:cut] + self.ellipsis_marker

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(tools)
        for tool in result:
            func = tool.get("function", {})
            if "description" in func:
                func["description"] = self._truncate(
                    func["description"], self.max_tool_desc_length
                )
            props = func.get("parameters", {}).get("properties", {})
            for _param_name, param_spec in props.items():
                if "description" in param_spec:
                    param_spec["description"] = self._truncate(
                        param_spec["description"], self.max_param_desc_length
                    )
        return result


class MinimalSchemaSerializer(ToolSerializer):
    """Strip optional / cosmetic fields from tool schemas.

    Removes parameter descriptions, keeps only name + type + required info.
    This tests whether the model can use tools from structure alone.

    Args:
        keep_tool_description: If ``True``, the top-level function description
                               is preserved.  If ``False`` it is also removed.
        keep_param_types:      If ``True``, parameter ``"type"`` fields are
                               preserved.
    """

    def __init__(
        self,
        keep_tool_description: bool = True,
        keep_param_types: bool = True,
    ) -> None:
        self.keep_tool_description = keep_tool_description
        self.keep_param_types = keep_param_types

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(tools)
        for tool in result:
            func = tool.get("function", {})
            if not self.keep_tool_description:
                func.pop("description", None)
            props = func.get("parameters", {}).get("properties", {})
            for _param_name, param_spec in props.items():
                param_spec.pop("description", None)
                if not self.keep_param_types:
                    param_spec.pop("type", None)
        return result


class JSONSchemaAnnotationSerializer(ToolSerializer):
    """Inject extra JSON-Schema annotations into tool parameter specs.

    This can be used to add ``"examples"`` or ``"default"`` hints to parameters
    that help guide the model.

    Args:
        annotations: Mapping ``tool_name -> param_name -> {extra_fields}``.
                     Extra fields are merged into the parameter's JSON schema
                     object.
    """

    def __init__(
        self,
        annotations: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        self.annotations = annotations

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(tools)
        for tool in result:
            func = tool.get("function", {})
            tool_name = func.get("name", "")
            if tool_name not in self.annotations:
                continue
            param_annotations = self.annotations[tool_name]
            props = func.get("parameters", {}).get("properties", {})
            for param_name, extra_fields in param_annotations.items():
                if param_name in props:
                    props[param_name].update(extra_fields)
        return result


class DescriptionPrefixSerializer(ToolSerializer):
    """Prepend a fixed string to every tool's description.

    Useful for injecting an instruction like ``"IMPORTANT: always provide all
    required parameters."`` into every tool definition.

    Args:
        prefix: The string to prepend (a space is inserted between the prefix
                and the original description).
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(tools)
        for tool in result:
            func = tool.get("function", {})
            desc = func.get("description", "")
            func["description"] = f"{self.prefix} {desc}".strip()
        return result


class XMLToolSerializer(ToolSerializer):
    """Re-encode tool definitions as XML strings embedded in the description.

    The ``"parameters"`` key is replaced with a minimal placeholder, and the
    full specification is written into the ``"description"`` field as XML.
    This tests whether an LLM can parse tools from an XML format.

    The top-level structure still conforms to the OpenAI tool-calling schema so
    the API call itself remains valid.
    """

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(tools)
        for tool in result:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            xml_lines = [f"<tool name=\"{name}\">"]
            if desc:
                xml_lines.append(f"  <description>{desc}</description>")
            xml_lines.append("  <parameters>")
            for pname, pspec in props.items():
                req = "true" if pname in required else "false"
                ptype = pspec.get("type", "any")
                pdesc = pspec.get("description", "")
                xml_lines.append(
                    f"    <param name=\"{pname}\" type=\"{ptype}\""
                    f" required=\"{req}\">{pdesc}</param>"
                )
            xml_lines.append("  </parameters>")
            xml_lines.append("</tool>")

            # Embed the XML in the description so the model sees it.
            func["description"] = "\n".join(xml_lines)
            # Keep parameters valid but minimal (models still need them for
            # the function-calling mechanism).
        return result


class CompositeToolSerializer(ToolSerializer):
    """Chain multiple serializers together, applying them in order.

    Args:
        serializers: Ordered sequence of serializers to apply.
    """

    def __init__(self, serializers: Sequence[ToolSerializer]) -> None:
        self.serializers: list[ToolSerializer] = list(serializers)

    def serialize_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for serializer in self.serializers:
            tools = serializer.serialize_tools(tools)
        return tools
