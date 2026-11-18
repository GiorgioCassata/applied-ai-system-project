import json

import pytest

from src import agent_settings as settings
from src import llm_client
from src.llm_client import CLIENT_CHOICES, DEFAULT_CLIENT, OLLAMA_MODELS, OllamaClient, make_client


# --- the settings file ---------------------------------------------------------------

def test_no_file_means_the_default(tmp_path):
    assert settings.load_settings(tmp_path / "missing.json") == {"client": DEFAULT_CLIENT}


def test_a_saved_choice_is_loaded_again(tmp_path):
    path = tmp_path / "agent.json"
    assert settings.save_settings({"client": "mock"}, path) == {"client": "mock"}
    assert settings.load_settings(path) == {"client": "mock"}


@pytest.mark.parametrize("choice", list(CLIENT_CHOICES))
def test_every_offered_choice_survives_a_round_trip(tmp_path, choice):
    path = tmp_path / "agent.json"
    settings.save_settings({"client": choice}, path)
    assert settings.load_settings(path)["client"] == choice


def test_saving_an_unknown_choice_saves_the_default(tmp_path):
    path = tmp_path / "agent.json"
    assert settings.save_settings({"client": "skynet"}, path) == {"client": DEFAULT_CLIENT}
    assert json.loads(path.read_text()) == {"client": DEFAULT_CLIENT}
    assert settings.save_settings({}, path) == {"client": DEFAULT_CLIENT}


@pytest.mark.parametrize("content", [
    "not json at all",
    "",
    "[1, 2, 3]",
    "null",
    '{"client": 5}',
    '{"client": "skynet"}',
    '{"client": "ollama"}',  # an older setting from before models were listed
    "{}",
])
def test_a_bad_or_outdated_file_means_the_default(tmp_path, content):
    path = tmp_path / "agent.json"
    path.write_text(content)
    assert settings.load_settings(path) == {"client": DEFAULT_CLIENT}


def test_a_model_taken_off_the_list_means_the_default(tmp_path):
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"client": "ollama:a-model-nobody-listed"}))
    assert settings.load_settings(path) == {"client": DEFAULT_CLIENT}


def test_saving_creates_the_folder_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "deeper" / "agent.json"
    settings.save_settings({"client": "mock"}, path)
    assert [p.name for p in path.parent.iterdir()] == ["agent.json"]


def test_a_later_save_replaces_the_earlier_one(tmp_path):
    path = tmp_path / "agent.json"
    settings.save_settings({"client": "gemini"}, path)
    settings.save_settings({"client": "mock"}, path)
    assert settings.load_settings(path)["client"] == "mock"


# --- the choices -----------------------------------------------------------------------

def test_every_listed_ollama_model_is_a_choice():
    assert OLLAMA_MODELS, "list at least one Ollama model at the top of llm_client.py"
    for model in OLLAMA_MODELS:
        assert f"ollama:{model}" in CLIENT_CHOICES
    assert {"gemini", "mock"} <= set(CLIENT_CHOICES)


def test_the_default_is_a_choice_and_is_the_first_model():
    assert DEFAULT_CLIENT in CLIENT_CHOICES
    assert DEFAULT_CLIENT == f"ollama:{OLLAMA_MODELS[0]}"


def test_every_choice_has_a_label():
    assert all(isinstance(label, str) and label for label in CLIENT_CHOICES.values())


# --- making a client from a choice -----------------------------------------------------

def test_make_client_mock():
    assert isinstance(make_client("mock"), llm_client.MockClient)


def test_make_client_ollama_uses_the_chosen_model():
    client = make_client(f"ollama:{OLLAMA_MODELS[0]}")
    assert isinstance(client, OllamaClient) and client.model_name == OLLAMA_MODELS[0]
    other = make_client("ollama:some-other-model:7b")
    assert other.model_name == "some-other-model:7b"  # only the first colon is the prefix


def test_make_client_gemini_without_a_key_explains_itself(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_client("gemini")


def test_make_client_unknown_choice():
    with pytest.raises(RuntimeError, match="unknown"):
        make_client("skynet")


# --- the clients' message translation ---------------------------------------------------

def test_ollama_messages_keep_tool_calls_and_name_the_tool_that_answered():
    neutral = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "find_song", "arguments": {"title": "x"}}]},
        {"role": "tool", "name": "find_song", "content": "{}"},
    ]
    sent = OllamaClient._messages(neutral)
    assert sent[2]["tool_calls"] == [{"function": {"name": "find_song", "arguments": {"title": "x"}}}]
    assert sent[3]["tool_name"] == "find_song"
    assert "tool_calls" not in sent[1] and "tool_name" not in sent[1]


def test_mock_client_hands_out_scripted_responses_then_the_default():
    mock = llm_client.MockClient(["one", llm_client.LLMResponse(text="two")], default="fallback")
    assert [mock.chat([]).text for _ in range(3)] == ["one", "two", "fallback"]
    assert len(mock.calls) == 3 and mock.tools_given == [None, None, None]
    assert mock.complete("system", "user") == "fallback"
