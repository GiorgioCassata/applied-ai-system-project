"""The chat agent (no Streamlit in here).

MusicAgent runs one tool-calling loop per user message:

    1. send the conversation and the tool schemas to the model
    2. if it asked for tools, run them (agent_tools.run_tool), add their results to the
       conversation, and go back to 1
    3. when it answers in plain text, that is the reply

The model does its own planning (it can ask for several tools at once, and for more
after seeing the results), and the system prompt tells it to look ids up before using
them and to check a change when a tool's result leaves doubt. That is cheaper and more
reliable than separate PLAN / ACT / REFLECT requests: three requests per step triple the
latency, and a small local model asked to "plan" in free text tends to invent ids or
claim it already did things. Most tools already return what changed (the new position,
the new song count), so a separate verification pass is usually redundant.

Guard rails, all in code rather than left to the prompt:
    * at most MAX_STEPS model requests per message, then it must answer with what it has
    * an exact repeat of a tool call in the same turn is refused (the model is looping)
    * tool errors go back to the model as data, so it can correct itself
    * a failed model request gives a fixed, friendly reply (the conversation is untouched)
    * run() can be stopped from outside (the `cancelled` check) between steps

An agent turn can take many seconds, and the app redraws its page (re-runs its script)
on every click. Run inline, a click would cut the agent off half way, so the app runs it
in an AgentJob: a background thread that carries on whatever the page does, and which the
page checks for the result.
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import agent_tools
except ImportError:  # imported as part of the "src" package (tests)
    from src import agent_tools

# Directory where prompt templates live, relative to this file
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

MAX_STEPS = 6  # model requests for one user message
MAX_TURNS_KEPT = 4  # earlier user messages (with their tool traffic) kept as context
FALLBACK_REPLY = "Sorry, I couldn't reach my brain just now. Please try again in a moment."
EMPTY_REPLY = "I'm not sure what to do with that. Could you say it a different way?"
UNDONE_REPLY = "Sorry, I wasn't able to do that, so nothing was changed. Please try asking again."
GIVE_UP_REPLY = "I wasn't able to finish that. Here is where I got to; tell me how you'd like to go on."


# A reply that says something was done ("I've added it", "Added!", "has been queued") or is
# about to be ("I'll play it", "let me add that"). Such a reply is only true if an action
# tool succeeded this turn. "Let me know" and the like are not claims.
_DID = r"(?:added|queued|played|started|created|removed|deleted|renamed|moved|saved|regenerated|put|cleared)"
_WILL = r"(?:play|add|queue|put|create|make|remove|delete|rename|move|save|start|regenerate|clear|skip)"
CLAIM_RE = re.compile(
    rf"\bi(?:'ve|\s+have|'m|\s+am)?\s+(?:just\s+|now\s+|also\s+|already\s+)?{_DID}\b"
    rf"|\b(?:i'll|i\s+will|i'm\s+going\s+to|i\s+am\s+going\s+to|let\s+me|going\s+to)\s+(?:just\s+|go\s+ahead\s+and\s+)?{_WILL}\b"
    rf"|\b(?:has|have|was|were)\s+(?:been\s+)?{_DID}\b"
    rf"|^\W*(?:{_DID}|done|all\s+set|now\s+playing)\b",
    re.IGNORECASE | re.MULTILINE,
)

NUDGE = (
    "[Note from the app, not the listener] Your last reply says you did something, or are "
    "about to, but no action tool succeeded this turn, so nothing has been done. If the "
    "listener asked for something, call the tool for it now, in this response. If you "
    "cannot do it, say so plainly and why. Never say something was done unless a tool "
    "result said ok."
)


def claims_action(text: str) -> bool:
    """Whether a reply claims (or promises) that something was done."""
    return bool(CLAIM_RE.search(text or ""))


def _without_thinking(text: str) -> str:
    """The text without <think>...</think> blocks, which some local models put in their
    answers: the listener only ever sees the final reply, never the reasoning."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ folder."""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class AgentResult:
    """The outcome of one user message.

    reply:    what to show the listener
    messages: everything added to the conversation this turn (the user message, tool
              calls and results, the reply); the caller appends it to its history
    ok:       False when the model could not be reached or the loop ran out of steps
    """

    reply: str
    messages: List[Dict] = field(default_factory=list)
    ok: bool = True


class MusicAgent:
    """Answers chat messages by calling the tools in agent_tools.py through an LLM client.

    client: anything with chat(messages, tools) -> LLMResponse (see llm_client.py)
    ctx:    the agent_tools.AgentContext the tools work on
    """

    def __init__(self, client: Any, ctx: "agent_tools.AgentContext",
                 system_prompt: Optional[str] = None):
        self.client = client
        self.ctx = ctx
        self.system_prompt = system_prompt or _load_prompt("system_prompt.txt")
        self.logs: List[Dict[str, Any]] = []  # what happened, for debugging

    def run(
        self,
        user_text: str,
        history: Optional[List[Dict]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Handles one user message. `history` is the earlier messages (from earlier
        AgentResult.messages); it is trimmed to the last few turns, not changed.

        `cancelled` is asked before every model request and every tool call; once it
        says True the turn stops there (ok False, empty reply): the changes already
        made stay, nothing more is done."""
        cancelled = cancelled or (lambda: False)
        new: List[Dict] = [{"role": "user", "content": user_text}]
        messages = [{"role": "system", "content": self.system_prompt}] + _trim(history or []) + new
        tools = agent_tools.tool_schemas()
        seen_calls = set()
        acted = False  # has an action tool succeeded this turn?
        nudged = False  # has the model been told off for an unbacked claim?

        for step in range(MAX_STEPS):
            # On the last step no tools are offered, which forces a plain-text answer
            if cancelled():
                return AgentResult("", new, ok=False)
            last = step == MAX_STEPS - 1
            response = self.client.chat(messages, None if last else tools)
            if response.error:
                self._log("error", response.error)
                return AgentResult(FALLBACK_REPLY, new, ok=False)

            if not response.tool_calls:
                reply = _without_thinking(response.text)
                if claims_action(reply) and not acted:
                    # "I've added it" / "I'll play it" with nothing done: the model talked
                    # instead of calling the tool. Give it one chance to really do it (the
                    # false claim and this note stay out of the kept conversation), and if
                    # it still only talks, say plainly that nothing happened.
                    self._log("unbacked_claim", reply)
                    if not nudged and not last:
                        nudged = True
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({"role": "user", "content": NUDGE})
                        continue
                    reply = UNDONE_REPLY
                elif not reply:
                    reply = "Done." if acted else EMPTY_REPLY
                new.append({"role": "assistant", "content": reply})
                return AgentResult(reply, new)

            assistant = {
                "role": "assistant",
                "content": _without_thinking(response.text),
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            }
            new.append(assistant)
            messages.append(assistant)
            for call in response.tool_calls:
                if cancelled():
                    return AgentResult("", new, ok=False)
                key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                if key in seen_calls:
                    result = {"ok": False, "error": "You already made this exact call this turn. "
                              "Use its earlier result instead of repeating it."}
                else:
                    seen_calls.add(key)
                    result = agent_tools.run_tool(self.ctx, call.name, call.arguments)
                self._log("tool", {"name": call.name, "arguments": call.arguments, "result": result})
                if result.get("ok") and agent_tools.is_action(call.name):
                    acted = True
                tool_message = {"role": "tool", "name": call.name,
                                "content": json.dumps(result, default=str)}
                new.append(tool_message)
                messages.append(tool_message)

        # Only reachable if the model still asked for tools on the last (tool-less) step
        new.append({"role": "assistant", "content": GIVE_UP_REPLY})
        return AgentResult(GIVE_UP_REPLY, new, ok=False)

    def _log(self, kind: str, detail: Any) -> None:
        self.logs.append({"type": kind, "detail": detail})


class AgentJob:
    """One agent turn running on a background thread, so it carries on while the page
    is redrawn. The page keeps the job, asks `done`, and then reads `result`.

    work: called as work(cancelled) on the thread, returns an AgentResult; `cancelled`
          is the check to hand to MusicAgent.run. If it raises, the result is the
          fallback reply (and the exception is kept in `error`).
    """

    def __init__(self, work: Callable[[Callable[[], bool]], AgentResult]):
        self._work = work
        self._cancel = threading.Event()
        self._done = threading.Event()
        self.result: Optional[AgentResult] = None
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, attach: Optional[Callable[[threading.Thread], None]] = None) -> "AgentJob":
        """Starts the thread. `attach(thread)` is called first, from the caller's thread
        (the app uses it to give the thread Streamlit's session context)."""
        self.thread = threading.Thread(target=self._run, daemon=True, name="music-agent")
        if attach:
            attach(self.thread)
        self.thread.start()
        return self

    def cancel(self) -> None:
        """Asks the agent to stop at its next step; it never undoes what it has done."""
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def _run(self) -> None:
        try:
            self.result = self._work(self.cancelled)
        except BaseException as exc:  # a crash must still end the job, or the chat waits forever
            self.error = exc
            self.result = AgentResult(FALLBACK_REPLY, [], ok=False)
        finally:
            self._done.set()


def _trim(history: List[Dict]) -> List[Dict]:
    """The last MAX_TURNS_KEPT turns of the history. A turn starts at a user message, so a
    tool result is never separated from the call that produced it."""
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(starts) <= MAX_TURNS_KEPT:
        return list(history)
    return list(history[starts[-MAX_TURNS_KEPT]:])
