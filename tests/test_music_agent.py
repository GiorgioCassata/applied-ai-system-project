import json
import threading

import pytest

from src import music_agent as ma
from src.llm_client import LLMResponse, MockClient, ToolCall
from src.music_agent import MAX_STEPS, MAX_TURNS_KEPT, AgentJob, AgentResult, MusicAgent
from src.play_queue import PlayQueue


def call(name, **arguments):
    """A model response that asks for one tool."""
    return LLMResponse(tool_calls=[ToolCall(name, arguments)])


def make_agent(ctx, responses, prompt="You are a test DJ."):
    client = MockClient(responses)
    return MusicAgent(client, ctx, system_prompt=prompt), client


def roles(messages):
    return [m["role"] for m in messages]


# --- a plain answer --------------------------------------------------------------------

def test_a_plain_answer_is_the_reply(ctx):
    agent, client = make_agent(ctx, ["Hello, listener!"])
    result = agent.run("hi")
    assert result.reply == "Hello, listener!" and result.ok
    assert roles(result.messages) == ["user", "assistant"]
    assert result.messages[0]["content"] == "hi"


def test_the_model_gets_the_system_prompt_the_message_and_the_tools(ctx):
    agent, client = make_agent(ctx, ["ok"], prompt="You are a test DJ.")
    agent.run("hi")
    sent = client.calls[0]
    assert sent[0] == {"role": "system", "content": "You are a test DJ."}
    assert sent[-1] == {"role": "user", "content": "hi"}
    names = {t["function"]["name"] for t in client.tools_given[0]}
    assert "find_song" in names and "add_song_to_queue" in names


def test_the_default_prompt_is_loaded_from_the_prompts_folder(ctx):
    agent = MusicAgent(MockClient(), ctx)
    assert "Wabi-Sabi" in agent.system_prompt


# --- using tools -----------------------------------------------------------------------

def test_a_tool_call_runs_and_its_result_goes_back_to_the_model(ctx):
    agent, client = make_agent(ctx, [call("find_song", title="song 17"), "Found Song 17."])
    result = agent.run("find song 17")

    assert result.reply == "Found Song 17." and result.ok
    assert roles(result.messages) == ["user", "assistant", "tool", "assistant"]
    asked = result.messages[1]["tool_calls"][0]
    assert asked == {"name": "find_song", "arguments": {"title": "song 17"}}
    answer = result.messages[2]
    assert answer["name"] == "find_song"
    assert json.loads(answer["content"])["songs"][0]["id"] == 17
    # the second request carried the tool's result
    assert client.calls[1][-1] == answer
    assert len(client.calls) == 2


def test_only_the_final_reply_is_a_reply(ctx):
    agent, _ = make_agent(ctx, [
        LLMResponse(text="Let me look that up...", tool_calls=[ToolCall("find_song", {"title": "song 5"})]),
        "Here it is.",
    ])
    result = agent.run("find song 5")
    assert result.reply == "Here it is."  # not the narration, not the tool's JSON


def test_several_tool_calls_in_one_response_all_run_in_order(ctx):
    both = LLMResponse(tool_calls=[ToolCall("find_song", {"title": "song 5"}), ToolCall("read_song", {"song_id": 5})])
    agent, _ = make_agent(ctx, [both, "Here you go."])
    result = agent.run("x")
    assert roles(result.messages) == ["user", "assistant", "tool", "tool", "assistant"]
    assert [m["name"] for m in result.messages if m["role"] == "tool"] == ["find_song", "read_song"]


def test_a_tool_can_change_the_app(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    agent, _ = make_agent(ctx, [call("add_song_to_queue", song_id=7), "Queued it."])
    agent.run("queue song 7")
    assert queue.up_next == [2, 3, 7]


def test_a_tool_error_goes_back_to_the_model_so_it_can_correct_itself(ctx):
    agent, client = make_agent(ctx, [call("read_song", song_id=999), call("read_song", song_id=5), "Got it."])
    result = agent.run("x")
    first, second = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
    assert first["ok"] is False and second["ok"] is True
    assert result.reply == "Got it." and result.ok


def test_an_unknown_tool_name_is_an_error_result_not_a_crash(ctx):
    agent, _ = make_agent(ctx, [call("make_coffee"), "Sorry, I can't."])
    result = agent.run("x")
    tool_result = json.loads(next(m for m in result.messages if m["role"] == "tool")["content"])
    assert tool_result["ok"] is False and "make_coffee" in tool_result["error"]
    assert result.reply == "Sorry, I can't."


def test_repeating_an_identical_call_in_one_turn_is_refused(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    same = call("add_song_to_queue", song_id=7)
    agent, _ = make_agent(ctx, [same, same, "Done."])
    result = agent.run("x")
    results = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False and "already made" in results[1]["error"]
    assert queue.up_next == [2, 3, 7]  # it ran once


# --- thinking never reaches the reply ------------------------------------------------

def test_think_blocks_are_stripped_from_the_reply(ctx):
    agent, _ = make_agent(ctx, ["<think>the user wants hello\nso...</think>Hello there!"])
    result = agent.run("hi")
    assert result.reply == "Hello there!"
    assert "think" not in result.messages[-1]["content"]


def test_a_think_block_with_a_missing_opening_tag_is_stripped_too(ctx):
    agent, _ = make_agent(ctx, ["reasoning about it...</think>Hi!"])
    assert agent.run("hi").reply == "Hi!"


def test_think_blocks_are_stripped_from_the_text_kept_beside_tool_calls(ctx):
    response = LLMResponse(text="<think>plan</think>", tool_calls=[ToolCall("find_song", {"title": "song 5"})])
    agent, _ = make_agent(ctx, [response, "ok"])
    result = agent.run("x")
    assert result.messages[1]["content"] == ""


def test_an_answer_that_is_only_thinking_does_not_pretend_something_was_done(ctx):
    agent, _ = make_agent(ctx, ["<think>hmm</think>"])
    assert agent.run("x").reply == ma.EMPTY_REPLY  # not "Done.": nothing was


def test_an_empty_answer_after_a_successful_action_is_done(ctx, queue_box):
    queue_box["queue"] = PlayQueue([1, 2, 3], 0, repeat="off")
    agent, _ = make_agent(ctx, [call("add_song_to_queue", song_id=7), ""])
    assert agent.run("x").reply == "Done."


# --- when things go wrong --------------------------------------------------------------

def test_a_failed_request_gives_the_fallback_reply(ctx):
    agent, _ = make_agent(ctx, [LLMResponse(error="ConnectionError: server down")])
    result = agent.run("hi")
    assert result.reply == ma.FALLBACK_REPLY and result.ok is False
    assert roles(result.messages) == ["user"]
    assert agent.logs[-1] == {"type": "error", "detail": "ConnectionError: server down"}


def test_a_failure_after_a_tool_call_still_gives_the_fallback(ctx):
    agent, _ = make_agent(ctx, [call("find_song", title="song 5"), LLMResponse(error="boom")])
    result = agent.run("x")
    assert result.reply == ma.FALLBACK_REPLY and result.ok is False


def test_the_loop_stops_after_max_steps_and_the_last_request_offers_no_tools(ctx):
    endless = [call("find_song", title=f"song {i}") for i in range(MAX_STEPS + 3)]
    agent, client = make_agent(ctx, endless)
    result = agent.run("x")

    assert len(client.calls) == MAX_STEPS
    assert all(t for t in client.tools_given[:-1]) and client.tools_given[-1] is None
    assert result.reply == ma.GIVE_UP_REPLY and result.ok is False
    assert result.messages[-1] == {"role": "assistant", "content": ma.GIVE_UP_REPLY}


def test_the_model_can_still_answer_on_the_last_step(ctx):
    steps = [call("find_song", title=f"song {i}") for i in range(MAX_STEPS - 1)] + ["Finally."]
    agent, client = make_agent(ctx, steps)
    result = agent.run("x")
    assert result.reply == "Finally." and result.ok
    assert client.tools_given[-1] is None


def test_tool_calls_and_results_are_logged(ctx):
    agent, _ = make_agent(ctx, [call("read_song", song_id=5), "ok"])
    agent.run("x")
    logged = [l for l in agent.logs if l["type"] == "tool"]
    assert logged[0]["detail"]["name"] == "read_song"
    assert logged[0]["detail"]["result"]["ok"] is True


# --- remembering the conversation ------------------------------------------------------

def test_earlier_turns_are_sent_to_the_model_and_the_history_is_not_changed(ctx):
    agent, client = make_agent(ctx, [call("find_song", title="song 17"), "It is id 17.", "Sure thing."])
    first = agent.run("find song 17")
    history = list(first.messages)

    second = agent.run("add it to my queue", history)
    sent = client.calls[2]  # the first request of the second turn
    assert sent[1:1 + len(history)] == history
    assert sent[-1] == {"role": "user", "content": "add it to my queue"}
    assert history == first.messages  # the caller's list is untouched
    assert roles(second.messages) == ["user", "assistant"]  # only the new turn comes back


def test_only_the_last_few_turns_are_kept(ctx):
    history = []
    for i in range(MAX_TURNS_KEPT + 4):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    agent, client = make_agent(ctx, ["ok"])
    agent.run("now", history)

    users = [m["content"] for m in client.calls[0] if m["role"] == "user"]
    assert len(users) == MAX_TURNS_KEPT + 1
    assert users[0] == f"q{4}" and users[-1] == "now"
    assert len(history) == 2 * (MAX_TURNS_KEPT + 4)


def test_trimming_never_separates_a_tool_result_from_its_call(ctx):
    turn = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "read_song", "arguments": {"song_id": 1}}]},
        {"role": "tool", "name": "read_song", "content": "{}"},
        {"role": "assistant", "content": "a"},
    ]
    history = turn * (MAX_TURNS_KEPT + 2)
    kept = ma._trim(history)
    assert roles(kept[:4]) == ["user", "assistant", "tool", "assistant"]
    assert len(kept) == 4 * MAX_TURNS_KEPT


def test_short_histories_are_kept_whole():
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert ma._trim(history) == history
    assert ma._trim([]) == []


# --- stopping an agent ---------------------------------------------------------------

def test_a_cancelled_agent_does_nothing(ctx):
    agent, client = make_agent(ctx, ["never said"])
    result = agent.run("hi", cancelled=lambda: True)
    assert result.reply == "" and result.ok is False
    assert client.calls == []  # not even a request to the model


def test_cancelling_between_tool_calls_stops_before_the_next_one(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    two = LLMResponse(tool_calls=[
        ToolCall("add_song_to_queue", {"song_id": 7}),
        ToolCall("add_song_to_queue", {"song_id": 8}),
    ])
    agent, client = make_agent(ctx, [two, "done"])
    checks = []

    def cancelled():
        checks.append(1)
        return len(checks) > 2  # fine before the request and before the first tool, then stop

    result = agent.run("x", cancelled=cancelled)
    assert result.ok is False and result.reply == ""
    assert queue.up_next == [2, 3, 7]  # what was done stays done; the second add never ran
    assert len(client.calls) == 1


# --- running in the background ---------------------------------------------------------

def finished(job):
    job.thread.join(timeout=5)
    assert job.done
    return job.result


def test_a_job_runs_the_work_on_another_thread_and_keeps_the_result():
    caller = threading.current_thread()
    seen = {}

    def work(cancelled):
        seen["thread"] = threading.current_thread()
        return AgentResult("hello")

    job = AgentJob(work)
    assert not job.done and job.result is None
    assert job.start() is job
    assert finished(job).reply == "hello"
    assert seen["thread"] is not caller and job.thread.daemon


def test_a_job_does_not_finish_until_the_work_does():
    release = threading.Event()
    job = AgentJob(lambda cancelled: (release.wait(5), AgentResult("late"))[1]).start()
    assert not job.done  # still working: a page redraw here would find no reply yet
    release.set()
    assert finished(job).reply == "late"


def test_the_work_is_given_the_cancel_check():
    started, proceed = threading.Event(), threading.Event()
    answers = []

    def work(cancelled):
        started.set()
        proceed.wait(5)
        answers.append(cancelled())
        return AgentResult("")

    job = AgentJob(work).start()
    started.wait(5)
    assert not job.cancelled()
    job.cancel()
    proceed.set()
    finished(job)
    assert job.cancelled() and answers == [True]


def test_a_crashing_job_still_finishes_with_the_fallback_reply():
    def work(cancelled):
        raise RuntimeError("boom")

    job = AgentJob(work).start()
    result = finished(job)
    assert result.reply == ma.FALLBACK_REPLY and result.ok is False
    assert isinstance(job.error, RuntimeError)


def test_attach_is_called_with_the_thread_before_it_starts():
    calls = []
    job = AgentJob(lambda cancelled: AgentResult("ok"))
    job.start(attach=lambda thread: calls.append((thread, thread.is_alive())))
    finished(job)
    assert calls == [(job.thread, False)]


def test_a_whole_agent_turn_runs_as_a_job_and_cancels(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    agent, _ = make_agent(ctx, [call("add_song_to_queue", song_id=7), "Queued."])
    job = AgentJob(lambda cancelled: agent.run("queue 7", [], cancelled)).start()
    result = finished(job)
    assert result.reply == "Queued." and queue.up_next == [2, 3, 7]


# --- a reply must not claim what the tools did not do --------------------------------

CLAIMS = [
    "I've added it to your queue.", "I added Song 5.", "Added!", "Now playing Hello.", "Done.",
    "I'll play that now.", "Let me add that for you.", "Song 5 has been queued.",
    "I am going to create the playlist.", "All set!", "Sure! I'll go ahead and queue it.",
    "I have already removed it.",
]
NOT_CLAIMS = [
    "Here are the songs I found.", "Up next is Song 2.", "Let me know what you'd like.",
    "I can add it if you like.", "Which album did you mean?", "I found two versions.",
    "You have 3 playlists.", "I couldn't add it because nothing is playing.",
    "Sorry, I wasn't able to do that.", "That song is already in the queue.", "",
]


@pytest.mark.parametrize("text", CLAIMS)
def test_claims_of_action_are_recognised(text):
    assert ma.claims_action(text)


@pytest.mark.parametrize("text", NOT_CLAIMS)
def test_ordinary_replies_are_not_claims(text):
    assert not ma.claims_action(text)


def test_a_claim_with_no_tool_call_is_challenged_and_the_model_gets_to_really_do_it(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    agent, client = make_agent(ctx, ["Sure, I'll add that now!", call("add_song_to_queue", song_id=7), "It's in your queue."])
    result = agent.run("add song 7")

    assert queue.up_next == [2, 3, 7]  # it really happened
    assert result.reply == "It's in your queue." and result.ok
    assert len(client.calls) == 3
    challenge = client.calls[1][-2:]  # what the model saw the second time
    assert challenge[0] == {"role": "assistant", "content": "Sure, I'll add that now!"}
    assert challenge[1]["role"] == "user" and "no action tool succeeded" in challenge[1]["content"]
    # neither the false claim nor the note is kept in the conversation
    assert roles(result.messages) == ["user", "assistant", "tool", "assistant"]
    assert not any("I'll add that now" in str(m.get("content")) for m in result.messages)
    assert [l["type"] for l in agent.logs if l["type"] == "unbacked_claim"] == ["unbacked_claim"]


def test_a_claim_that_is_repeated_becomes_an_honest_reply(ctx, queue_box):
    queue = PlayQueue([1, 2, 3], 0, repeat="off")
    queue_box["queue"] = queue
    agent, client = make_agent(ctx, ["I've added it!", "I have added it, honestly."])
    result = agent.run("add song 7")
    assert result.reply == ma.UNDONE_REPLY and result.ok
    assert queue.up_next == [2, 3]  # nothing changed, and the reply says so
    assert len(client.calls) == 2  # challenged once, not endlessly
    assert result.messages[-1] == {"role": "assistant", "content": ma.UNDONE_REPLY}


def test_a_claim_after_a_failed_action_is_challenged(ctx):
    # nothing is playing, so the queue tool fails; the model then claims success anyway
    agent, client = make_agent(ctx, [call("add_song_to_queue", song_id=7), "Added it!", "Sorry, nothing is playing."])
    result = agent.run("queue song 7")
    assert result.reply == "Sorry, nothing is playing."
    assert len(client.calls) == 3


def test_a_claim_backed_by_a_successful_action_is_left_alone(ctx, queue_box):
    queue_box["queue"] = PlayQueue([1, 2, 3], 0, repeat="off")
    agent, client = make_agent(ctx, [call("add_song_to_queue", song_id=7), "I've added it to your queue."])
    assert agent.run("x").reply == "I've added it to your queue."
    assert len(client.calls) == 2


def test_a_claim_after_only_looking_things_up_is_challenged(ctx):
    agent, client = make_agent(ctx, [call("find_song", title="song 5"), "I've queued it.", "I found it but have not queued it."])
    result = agent.run("x")
    assert len(client.calls) == 3 and result.reply == "I found it but have not queued it."


def test_plain_answers_and_lookups_are_never_challenged(ctx):
    agent, client = make_agent(ctx, ["Which song do you mean?"])
    assert agent.run("play something").reply == "Which song do you mean?"
    agent, client = make_agent(ctx, [call("find_song", title="song 5"), "Song 5 is by Artist 1."])
    assert agent.run("who sings song 5").reply == "Song 5 is by Artist 1."
    assert len(client.calls) == 2


def test_no_challenge_is_possible_on_the_last_step(ctx):
    steps = [call("find_song", title=f"song {i}") for i in range(MAX_STEPS - 1)] + ["I've added it!"]
    agent, client = make_agent(ctx, steps)
    result = agent.run("x")  # no tools are offered on the last step, so it cannot act: say so
    assert result.reply == ma.UNDONE_REPLY and len(client.calls) == MAX_STEPS
