import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import Mock

import pytest

import core


def test_gemini_provider_sends_chat_messages(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(core.requests, "post", post)

    result = core.llm([{"role": "user", "content": "Say hello"}], max_tokens=50)

    assert result == "Hello from Gemini"
    args, kwargs = post.call_args
    assert "gemini-test-model:generateContent" in args[0]
    assert kwargs["params"]["key"] == "test-gemini-key"
    assert kwargs["json"]["contents"][0]["parts"][0]["text"] == "Say hello"


def test_groq_provider_sends_openai_compatible_chat(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-test-model")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "Hello from Groq"}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(core.requests, "post", post)

    result = core.llm([{"role": "user", "content": "Say hello"}], max_tokens=50)

    assert result == "Hello from Groq"
    args, kwargs = post.call_args
    assert args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-groq-key"
    assert kwargs["json"]["model"] == "groq-test-model"


def test_local_transcription_uses_faster_whisper(monkeypatch, tmp_path):
    calls = {}

    class FakeModel:
        def __init__(self, model_size, device, compute_type):
            calls["init"] = (model_size, device, compute_type)

        def transcribe(self, audio_path, beam_size, vad_filter):
            calls["transcribe"] = (audio_path, beam_size, vad_filter)
            return iter([types.SimpleNamespace(text=" First sentence."),
                         types.SimpleNamespace(text=" Second sentence.")]), types.SimpleNamespace(language="en")

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setenv("WHISPER_MODEL", "base")
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio")

    result = core.transcribe_audio(audio)

    assert result == "First sentence. Second sentence."
    assert calls["init"] == ("base", "cpu", "int8")
    assert calls["transcribe"] == (str(audio), 5, True)


def test_missing_selected_provider_key_has_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        core.llm([{"role": "user", "content": "hello"}])


def test_groq_rate_limit_explains_free_quota_without_payment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    error_response = Mock()
    error_response.status_code = 429
    response = Mock()
    response.raise_for_status.side_effect = __import__("requests").HTTPError(
        "rate limited", response=error_response
    )
    monkeypatch.setattr(core.requests, "post", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="free-tier rate limit.*no payment is required"):
        core.llm([{"role": "user", "content": "hello"}])
