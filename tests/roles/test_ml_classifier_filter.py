"""Tests for MLClassifierToolFilter and end-to-end agent config integration."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from unittest.mock import patch

import pytest

from tool_sandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    set_current_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.tool_filter import MLClassifierToolFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _execution_context() -> None:
    ctx = ExecutionContext()
    set_current_context(ctx)


def _make_dummy_tool(name: str, doc: str = "dummy") -> Callable[..., Any]:
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
# Mock HTTP server for classifier endpoint
# ---------------------------------------------------------------------------


class _MockClassifierHandler(BaseHTTPRequestHandler):
    """Returns pre-configured scores for /predict."""

    scores: list[float] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        # Store the request body for assertions
        _MockClassifierHandler.last_request = body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = json.dumps({"scores": _MockClassifierHandler.scores})
        self.wfile.write(resp.encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress output


@pytest.fixture()
def mock_classifier():
    """Start a local HTTP server that mimics the classifier endpoint."""
    server = HTTPServer(("127.0.0.1", 0), _MockClassifierHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/predict", _MockClassifierHandler
    server.shutdown()


# ---------------------------------------------------------------------------
# MLClassifierToolFilter tests
# ---------------------------------------------------------------------------


class TestMLClassifierToolFilter:
    """Tests for the ML classifier-based tool filter."""

    def test_filter_by_threshold(self, mock_classifier: tuple) -> None:
        """Only tools above the threshold are kept."""
        url, handler = mock_classifier
        handler.scores = [0.9, 0.3, 0.8, 0.1]

        filt = MLClassifierToolFilter(endpoint_url=url, threshold=0.5)
        tools = _make_tools()
        messages = _make_messages()

        result = filt.filter_tools(tools, messages)
        names = set(result.keys())
        # search_contacts=0.9 and set_wifi=0.8 pass; send_message=0.3 and
        # add_reminder=0.1 do not.
        assert names == {"search_contacts", "set_wifi"}

    def test_threshold_zero_keeps_all(self, mock_classifier: tuple) -> None:
        """With threshold=0, every tool is included."""
        url, handler = mock_classifier
        handler.scores = [0.01, 0.02, 0.03, 0.04]

        filt = MLClassifierToolFilter(endpoint_url=url, threshold=0.0)
        result = filt.filter_tools(_make_tools(), _make_messages())
        assert len(result) == 4

    def test_always_include(self, mock_classifier: tuple) -> None:
        """always_include tools bypass the threshold."""
        url, handler = mock_classifier
        handler.scores = [0.1, 0.1, 0.1, 0.1]

        filt = MLClassifierToolFilter(
            endpoint_url=url,
            threshold=0.5,
            always_include=["add_reminder", "set_wifi"],
        )
        result = filt.filter_tools(_make_tools(), _make_messages())
        assert set(result.keys()) == {"add_reminder", "set_wifi"}

    def test_always_include_overlaps_with_scored(
        self, mock_classifier: tuple
    ) -> None:
        url, handler = mock_classifier
        handler.scores = [0.9, 0.1, 0.1, 0.1]

        filt = MLClassifierToolFilter(
            endpoint_url=url,
            threshold=0.5,
            always_include=["add_reminder"],
        )
        result = filt.filter_tools(_make_tools(), _make_messages())
        # search_contacts via score, add_reminder via always_include
        assert set(result.keys()) == {"search_contacts", "add_reminder"}

    def test_empty_tools(self, mock_classifier: tuple) -> None:
        url, _ = mock_classifier
        filt = MLClassifierToolFilter(endpoint_url=url)
        result = filt.filter_tools({}, _make_messages())
        assert result == {}

    def test_fallback_on_connection_error(self) -> None:
        """When the classifier endpoint is unreachable, return ALL tools."""
        filt = MLClassifierToolFilter(
            endpoint_url="http://127.0.0.1:1/predict",
            threshold=0.5,
            timeout=0.5,
        )
        tools = _make_tools()
        result = filt.filter_tools(tools, _make_messages())
        assert result == tools  # all tools returned as fallback

    def test_request_body_structure(self, mock_classifier: tuple) -> None:
        """Verify the JSON body sent to the classifier endpoint."""
        url, handler = mock_classifier
        handler.scores = [0.5, 0.5, 0.5, 0.5]

        filt = MLClassifierToolFilter(endpoint_url=url, threshold=0.5)
        filt.filter_tools(_make_tools(), _make_messages("Turn off wifi"))

        body = handler.last_request
        assert body["user_message"] == "Turn off wifi"
        assert len(body["tool_names"]) == 4
        assert len(body["tool_descriptions"]) == 4
        assert "search_contacts" in body["tool_names"]

    def test_uses_last_user_message(self, mock_classifier: tuple) -> None:
        """The filter should use the most recent USER→AGENT message."""
        url, handler = mock_classifier
        handler.scores = [0.5, 0.5, 0.5, 0.5]

        filt = MLClassifierToolFilter(endpoint_url=url, threshold=0.5)
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="First message",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Agent response",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Second message",
            ),
        ]
        filt.filter_tools(_make_tools(), messages)
        assert handler.last_request["user_message"] == "Second message"


# ---------------------------------------------------------------------------
# Agent config integration
# ---------------------------------------------------------------------------


class TestMLClassifierAgentConfig:
    """Test that ml_classifier filter is correctly built from JSON config."""

    def test_build_ml_classifier_filter(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {
            "type": "ml_classifier",
            "endpoint_url": "http://my-host:9999/predict",
            "threshold": 0.7,
            "always_include": ["search_contacts"],
            "timeout": 5.0,
        }
        filt = _build_tool_filter(config)
        assert isinstance(filt, MLClassifierToolFilter)
        assert filt.endpoint_url == "http://my-host:9999/predict"
        assert filt.threshold == 0.7
        assert filt.always_include == {"search_contacts"}
        assert filt.timeout == 5.0

    def test_build_ml_classifier_defaults(self) -> None:
        from tool_sandbox.cli.agent_config import _build_tool_filter

        config = {"type": "ml_classifier"}
        filt = _build_tool_filter(config)
        assert isinstance(filt, MLClassifierToolFilter)
        assert filt.endpoint_url == "http://localhost:5050/predict"
        assert filt.threshold == 0.5
        assert filt.always_include == set()

    def test_full_agent_config_with_ml_classifier(self, tmp_path: Any) -> None:
        """A full agent config JSON with tool_filtered + ml_classifier loads."""
        from tool_sandbox.cli.agent_config import build_agent_from_config

        config = {
            "type": "tool_filtered",
            "inner_agent": "Unhelpful",
            "tool_filter": {
                "type": "ml_classifier",
                "endpoint_url": "http://localhost:5050/predict",
                "threshold": 0.6,
            },
        }
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config))

        # build_agent_from_config expects a dict, not a file path
        agent = build_agent_from_config(config)
        from tool_sandbox.roles.tool_filtered_agent import ToolFilteredAgent

        assert isinstance(agent, ToolFilteredAgent)
        assert isinstance(agent.tool_filter, MLClassifierToolFilter)
        assert agent.tool_filter.threshold == 0.6


# ---------------------------------------------------------------------------
# Classifier project module tests
# ---------------------------------------------------------------------------


class TestToolClassifierModel:
    """Basic tests for the standalone classifier model (if importable)."""

    @pytest.fixture(autouse=True)
    def _check_deps(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")

    def test_model_instantiation(self) -> None:
        import sys
        from pathlib import Path

        # Add the tool_classifier project to path
        tc_root = (
            Path(__file__).resolve().parent.parent.parent.parent / "tool_classifier"
        )
        if str(tc_root) not in sys.path:
            sys.path.insert(0, str(tc_root))

        from tool_classifier.model import ToolRelevanceClassifier

        model = ToolRelevanceClassifier(hidden_dim=32)
        scores = model.predict_proba(
            ["Turn off wifi"],
            ["set_wifi_status"],
            ["Toggle WiFi on or off"],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_save_and_load(self, tmp_path: Any) -> None:
        import sys
        from pathlib import Path

        tc_root = (
            Path(__file__).resolve().parent.parent.parent.parent / "tool_classifier"
        )
        if str(tc_root) not in sys.path:
            sys.path.insert(0, str(tc_root))

        from tool_classifier.model import ToolRelevanceClassifier

        model = ToolRelevanceClassifier(hidden_dim=32)
        save_path = tmp_path / "model.pt"
        model.save(save_path)
        assert save_path.exists()

        loaded = ToolRelevanceClassifier.load(save_path)
        scores = loaded.predict_proba(
            ["Turn off wifi"],
            ["set_wifi_status"],
            ["Toggle WiFi on or off"],
        )
        assert len(scores) == 1
