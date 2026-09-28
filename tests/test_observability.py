"""
Tests for the Langfuse observability helpers
============================================

Tracing must be a transparent no-op when disabled and use the Langfuse
observation API when a client is available.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from memorytwin import observability


def test_tracing_is_disabled_under_pytest():
    assert observability._is_disabled() is True


def test_trace_observation_yields_none_when_disabled():
    with observability.trace_observation("Anything") as observation:
        assert observation is None


def test_trace_observation_uses_observation_api():
    observation = MagicMock()
    client = MagicMock()

    @contextmanager
    def fake_start(**kwargs):
        yield observation

    client.start_as_current_observation.side_effect = fake_start

    with patch.object(observability, "_is_disabled", return_value=False), \
            patch.object(observability, "_get_langfuse", return_value=client):
        with observability.trace_observation("LLM call", as_type="generation", model="m") as obs:
            obs.update(output="done")

    client.start_as_current_observation.assert_called_once_with(as_type="generation", name="LLM call", model="m")
    observation.update.assert_called_once_with(output="done")
    client.flush.assert_called_once()


@pytest.mark.asyncio
async def test_decorators_are_transparent_when_disabled():
    @observability.trace_access_memory
    async def query(self, question):
        return {"answer": question.upper()}

    assert await query(None, "why?") == {"answer": "WHY?"}
