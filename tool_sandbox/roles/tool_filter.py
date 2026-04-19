"""Tool filtering infrastructure for customising which tools an agent can see.

A :class:`ToolFilter` is an optional component that sits between the execution
context's full tool set and the agent.  When attached to a
:class:`~tool_sandbox.roles.base_role.BaseRole` instance (via ``_tool_filter``),
the filter's :meth:`filter_tools` method is called each turn before the agent
invokes the LLM.  This makes it possible to:

* Restrict tools to a dynamically-computed subset (e.g. based on conversation
  context).
* Use a classifier (possibly an LLM) to decide which tools are relevant for
  the current user request.
* Compose multiple filters together.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, Optional, Sequence

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message

LOGGER = getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ToolFilter(ABC):
    """Base class for all tool filters.

    Subclasses must implement :meth:`filter_tools` which receives the full set
    of available tools together with the current conversation and returns the
    (possibly reduced) subset the agent should see.
    """

    @abstractmethod
    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        """Return the subset of *tools* that should be exposed to the agent.

        Args:
            tools:    All tools currently visible to the agent (name -> callable).
            messages: The conversation history so far.

        Returns:
            A (possibly strict) subset of *tools*.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class AllowListToolFilter(ToolFilter):
    """Keep only tools whose names appear in an explicit allow-list.

    This is the simplest possible filter – useful as a baseline or to hard-code
    a fixed tool subset for a particular agent in a multi-agent setup.

    Args:
        allowed_tool_names: Names of tools to keep.
    """

    def __init__(self, allowed_tool_names: Sequence[str]) -> None:
        self.allowed_tool_names: set[str] = set(allowed_tool_names)

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        return {
            name: tool
            for name, tool in tools.items()
            if name in self.allowed_tool_names
        }


class DenyListToolFilter(ToolFilter):
    """Remove tools whose names appear in a deny-list.

    Args:
        denied_tool_names: Names of tools to remove.
    """

    def __init__(self, denied_tool_names: Sequence[str]) -> None:
        self.denied_tool_names: set[str] = set(denied_tool_names)

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        return {
            name: tool
            for name, tool in tools.items()
            if name not in self.denied_tool_names
        }


class KeywordToolFilter(ToolFilter):
    """Filter tools based on keyword matches in the most recent user message.

    The filter maintains a mapping from regex patterns to tool name sets.  If
    the last user message matches a pattern the corresponding tools are
    included.  Tools not associated with any matched pattern are excluded
    unless ``include_unmatched`` is *True*.

    Args:
        keyword_to_tools:  Mapping from regex pattern string to a list of tool
                           names that should be included when the pattern
                           matches.
        include_unmatched: If *True*, tools that are not mentioned in any
                           pattern value list are always included. Defaults to
                           *False*.
        default_tools:     Tool names that are always included regardless of
                           keyword matching (e.g. utility tools).
    """

    def __init__(
        self,
        keyword_to_tools: dict[str, list[str]],
        include_unmatched: bool = False,
        default_tools: Optional[Sequence[str]] = None,
    ) -> None:
        self.keyword_to_tools = {
            re.compile(pattern, re.IGNORECASE): tool_names
            for pattern, tool_names in keyword_to_tools.items()
        }
        self.include_unmatched = include_unmatched
        self.default_tools: set[str] = set(default_tools or [])

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        # Find the last user-sent message content
        last_user_content = ""
        for msg in reversed(messages):
            if msg.sender is not None and msg.sender == RoleType.USER:
                last_user_content = msg.content or ""
                break

        matched_tool_names: set[str] = set(self.default_tools)
        any_pattern_matched = False
        for pattern, tool_names in self.keyword_to_tools.items():
            if pattern.search(last_user_content):
                matched_tool_names.update(tool_names)
                any_pattern_matched = True

        if not any_pattern_matched and self.include_unmatched:
            return tools

        # Registered tools that are not in any keyword mapping
        all_mapped_tools: set[str] = set()
        for names_list in self.keyword_to_tools.values():
            all_mapped_tools.update(names_list)

        result: dict[str, Callable[..., Any]] = {}
        for name, tool in tools.items():
            if name in matched_tool_names:
                result[name] = tool
            elif self.include_unmatched and name not in all_mapped_tools:
                result[name] = tool
        return result


class LLMClassifierToolFilter(ToolFilter):
    """Use an LLM to classify which tools are relevant for the current turn.

    This sends a classification prompt to a (potentially cheap/fast) LLM that
    lists the available tools and the most recent user message, and asks the
    LLM to output a JSON array of relevant tool names.

    Args:
        model_name:     OpenAI-compatible model name for classification.
        api_key:        Optional API key.  Falls back to ``OPENAI_API_KEY`` env
                        var.
        base_url:       Optional base URL for the OpenAI-compatible endpoint.
        max_tools:      Maximum number of tools to return.  If the classifier
                        returns more, they are truncated (in the order the LLM
                        returned them).  ``None`` means no limit.
        always_include: Tool names that are always included regardless of what
                        the classifier says.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tools: Optional[int] = None,
        always_include: Optional[Sequence[str]] = None,
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
        self.max_tools = max_tools
        self.always_include: set[str] = set(always_include or [])

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        if not tools:
            return tools

        # Build the tool descriptions for the classifier prompt
        tool_descriptions: list[str] = []
        for name, tool in tools.items():
            doc = (getattr(tool, "__doc__", None) or "No description.").split("\n")[0]
            tool_descriptions.append(f"- {name}: {doc}")
        tool_list_str = "\n".join(tool_descriptions)

        # Extract conversation context (last few messages)
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        context_parts: list[str] = []
        for msg in recent_messages:
            sender = msg.sender.value if msg.sender else "unknown"
            content = (msg.content or "")[:500]
            context_parts.append(f"[{sender}]: {content}")
        conversation_context = "\n".join(context_parts)

        prompt = (
            "You are a tool-selection classifier. Given the conversation context "
            "and the list of available tools below, return a JSON array of tool "
            "names that are relevant for handling the current request. Return "
            "ONLY the JSON array, no other text.\n\n"
            "Available tools:\n"
            f"{tool_list_str}\n\n"
            "Conversation context:\n"
            f"{conversation_context}\n\n"
            "Relevant tool names (JSON array):"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "[]"
            # Parse the JSON array;  handle both ```json ... ``` and plain []
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            selected_names: list[str] = json.loads(content)
        except Exception:
            LOGGER.warning(
                "LLM classifier failed; falling back to returning all tools.",
                exc_info=True,
            )
            return tools

        selected_set: set[str] = set(selected_names) | self.always_include
        if self.max_tools is not None:
            # Prioritise tools the classifier returned, then always_include
            ordered = [n for n in selected_names if n in tools]
            for n in self.always_include:
                if n not in ordered and n in tools:
                    ordered.append(n)
            ordered = ordered[: self.max_tools]
            selected_set = set(ordered)

        return {
            name: tool for name, tool in tools.items() if name in selected_set
        }


class MLClassifierToolFilter(ToolFilter):
    """Use a trained ML binary classifier to predict tool relevance.

    For each (user_message, tool_name) pair the classifier produces a
    probability that the tool is relevant.  Tools whose score exceeds
    ``threshold`` are kept.

    The classifier is queried via an HTTP endpoint that accepts a JSON body::

        POST /predict
        {
            "user_message": "Turn off wifi",
            "tool_names": ["set_wifi_status", "search_contacts", ...],
            "tool_descriptions": ["Toggle the WiFi ...", "Search for ...", ...]
        }

    and returns::

        {"scores": [0.95, 0.02, ...]}

    This decoupling means the classifier can be a separate micro-service
    (PyTorch, TensorFlow, scikit-learn, etc.) running in its own container.

    Args:
        endpoint_url:   Full URL of the ``/predict`` endpoint.
        threshold:      Minimum score to keep a tool (default: 0.5).
        always_include: Tool names that bypass the classifier.
        timeout:        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:5050/predict",
        threshold: float = 0.5,
        always_include: Optional[Sequence[str]] = None,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.threshold = threshold
        self.always_include: set[str] = set(always_include or [])
        self.timeout = timeout

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        if not tools:
            return tools

        # Extract the most recent user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.sender is not None and msg.sender == RoleType.USER:
                last_user_content = msg.content or ""
                break

        # Build tool names and short descriptions
        tool_names: list[str] = []
        tool_descriptions: list[str] = []
        for name, tool in tools.items():
            tool_names.append(name)
            doc = (getattr(tool, "__doc__", None) or "").split("\n")[0]
            tool_descriptions.append(doc)

        payload = json.dumps(
            {
                "user_message": last_user_content,
                "tool_names": tool_names,
                "tool_descriptions": tool_descriptions,
            }
        ).encode("utf-8")

        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout
            ) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            scores: list[float] = body["scores"]
        except Exception:
            LOGGER.warning(
                "ML classifier request failed; falling back to all tools.",
                exc_info=True,
            )
            return tools

        # Select tools that pass the threshold or are always-included
        selected: dict[str, Callable[..., Any]] = {}
        for name, score in zip(tool_names, scores):
            if score >= self.threshold or name in self.always_include:
                selected[name] = tools[name]

        # Ensure always_include tools are present even if they weren't scored
        for name in self.always_include:
            if name in tools and name not in selected:
                selected[name] = tools[name]

        return selected


class CompositeToolFilter(ToolFilter):
    """Chain multiple filters in sequence (logical AND / pipeline).

    The output of each filter is fed as input to the next one.

    Args:
        filters: Ordered sequence of filters to apply.
    """

    def __init__(self, filters: Sequence[ToolFilter]) -> None:
        self.filters: list[ToolFilter] = list(filters)

    def filter_tools(
        self,
        tools: dict[str, Callable[..., Any]],
        messages: list[Message],
    ) -> dict[str, Callable[..., Any]]:
        for f in self.filters:
            tools = f.filter_tools(tools, messages)
        return tools
