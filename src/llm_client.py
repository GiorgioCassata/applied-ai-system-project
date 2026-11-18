"""LLM clients: one small interface over Gemini, a local Ollama model, and a scripted
mock for tests. The agent (music_agent.py) never knows which one it is talking to.

Every client has:

    chat(messages, tools) -> LLMResponse     what the agent uses (tool calling)
    complete(system_prompt, user_prompt)     a plain one-shot text answer

Messages are plain dicts in one neutral format, which each client translates:

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"name": str, "arguments": dict}],   # assistant messages only
     "name": str}                                        # tool messages: which tool answered

`tools` is the list of schemas from agent_tools.tool_schemas(). A failure (server down,
bad key, blocked reply) never raises: it comes back as LLMResponse.error, so the agent
can answer with a safe message instead of crashing the chat.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- The models to choose from: add yours here -----------------------------------------
# Local Ollama models, by the name `ollama list` shows (pull one first: `ollama pull <name>`).
# Each one appears in the Settings page's "Language model" list; the first is the default.
# Pick a model that supports tool calling, or the agent will not be able to use its tools.
OLLAMA_MODELS = [
    "gemma4:latest",
    "ornith:9b",
]
# ----------------------------------------------------------------------------------------

OLLAMA_PREFIX = "ollama:"

# What the Settings page offers, as {saved choice: label}. A choice is "gemini", "mock", or
# "ollama:<model name>" for each model in OLLAMA_MODELS.
CLIENT_CHOICES = {
    **{f"{OLLAMA_PREFIX}{m}": f"Ollama: {m} (local)" for m in OLLAMA_MODELS},
    "gemini": "Gemini (cloud, needs GEMINI_API_KEY)",
    "mock": "Mock (offline, scripted replies for testing)",
}
DEFAULT_CLIENT = next(iter(CLIENT_CHOICES))  # the first Ollama model (or Gemini if there is none)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """What the model said: text, and/or tools it wants called. `error` is set (and the
    rest empty) when the request failed."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    error: Optional[str] = None
    thinking: str = ""  # the model's reasoning trace, when it gives one (never shown to the listener)


class MockClient:
    """Offline stand-in for an LLM client, for tests and for running without a model.

    Give it the responses to hand out, in order (an LLMResponse, or a plain string for a
    text-only answer). Once they run out it answers with `default`. Every call's messages
    are kept in `calls`, and the tools each one offered in `tools_given`, so a test can
    check what the agent sent."""

    def __init__(self, responses: Optional[List[Any]] = None, default: str = "(mock reply)"):
        self.responses = list(responses or [])
        self.default = default
        self.calls: List[List[Dict]] = []
        self.tools_given: List[Optional[List[Dict]]] = []

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_given.append(tools)
        if not self.responses:
            return LLMResponse(text=self.default)
        nxt = self.responses.pop(0)
        return nxt if isinstance(nxt, LLMResponse) else LLMResponse(text=str(nxt))

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        ).text


class GeminiClient:
    """Gemini API wrapper.

    Requirements:
    - google-genai installed
    - GEMINI_API_KEY set in the environment (or loaded via python-dotenv)
    """

    def __init__(self, model_name: str = "gemini-flash-lite-latest", temperature: float = 0.2):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Create a .env file and set GEMINI_API_KEY=..."
            )
        # Imported here so other clients do not need the dependency.
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = float(temperature)

    def _contents(self, messages: List[Dict]):
        """(system instruction, Gemini contents) from the neutral message list."""
        types = self._types
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            role = m["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif role == "assistant":
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [
                    types.Part.from_function_call(name=c["name"], args=c.get("arguments") or {})
                    for c in m.get("tool_calls", [])
                ]
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                try:
                    payload = json.loads(m["content"])
                except ValueError:
                    payload = {"result": m["content"]}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=m.get("name", "tool"), response=payload)
                ]))
        return system, contents

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> LLMResponse:
        types = self._types
        try:
            system, contents = self._contents(messages)
            config = types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=self.temperature,
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["function"]["name"],
                        description=t["function"]["description"],
                        parameters=t["function"]["parameters"],
                    ) for t in tools
                ])] if tools else None,
                # The agent runs the tools itself, so Gemini must not try to
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            calls, texts = [], []
            parts = (response.candidates[0].content.parts or []) if response.candidates else []
            for part in parts:
                if getattr(part, "function_call", None):
                    calls.append(ToolCall(part.function_call.name, dict(part.function_call.args or {})))
                elif getattr(part, "text", None):
                    texts.append(part.text)
            return LLMResponse(text="".join(texts).strip(), tool_calls=calls)
        except Exception as exc:  # blocked reply, bad key, no network...
            return LLMResponse(error=f"{type(exc).__name__}: {exc}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One plain request; an empty string if anything goes wrong."""
        return self.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        ).text


def make_client(name: str):
    """The client for a choice in CLIENT_CHOICES. Raises RuntimeError with a message the
    listener can read when it cannot be made (no API key, package not installed)."""
    if name == "mock":
        return MockClient()
    try:
        if name == "gemini":
            return GeminiClient()
        if name.startswith(OLLAMA_PREFIX):
            return OllamaClient(model_name=name[len(OLLAMA_PREFIX):])
    except ImportError as exc:
        raise RuntimeError(f"a required package is not installed ({exc.name})") from exc
    raise RuntimeError(f"unknown client '{name}'")


class OllamaClient:
    """Local LLM wrapper using Ollama (no API key needed).

    Requirements:
    - `ollama` Python package installed
    - Ollama running locally with a tool-capable model pulled (e.g. `ollama pull gemma4`)
    """

    def __init__(self, model_name: str = OLLAMA_MODELS[0], temperature: float = 0.2):
        import ollama  # imported here so other clients do not need the package

        self.client = ollama.Client()
        self.model_name = model_name
        self.temperature = float(temperature)

    @staticmethod
    def _messages(messages: List[Dict]) -> List[Dict]:
        out = []
        for m in messages:
            item = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                item["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                    for c in m["tool_calls"]
                ]
            if m["role"] == "tool" and m.get("name"):
                item["tool_name"] = m["name"]
            out.append(item)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> LLMResponse:
        try:
            kwargs = dict(
                model=self.model_name,
                messages=self._messages(messages),
                tools=tools or None,
                options={"temperature": self.temperature},
            )
            try:
                response = self.client.chat(think=True, **kwargs)
            except Exception as exc:
                if "think" not in str(exc).lower():
                    raise
                response = self.client.chat(**kwargs)  # model has no thinking mode
            message = response["message"]
            calls = []
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):  # some models send the arguments as JSON text
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {}
                calls.append(ToolCall(call["function"]["name"], dict(arguments or {})))
            content = message.get("content") or ""
            thinking = (message.get("thinking") or "").strip()
            inline = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
            if inline:  # some models put the reasoning in the text instead
                thinking = "\n\n".join([t for t in [thinking, *(i.strip() for i in inline)] if t])
            return LLMResponse(text=content.strip(), tool_calls=calls, thinking=thinking)
        except Exception as exc:  # server down, model missing, model without tool support...
            return LLMResponse(error=f"{type(exc).__name__}: {exc}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One plain request; an empty string if anything goes wrong."""
        return self.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        ).text
