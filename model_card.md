# Model Card: Legally Distinct-ify and its AI DJ

This card covers the responsible-AI side of Legally Distinct-ify: the FlowFinder recommender that scores songs, and "Wabi-Sabi", the chat agent that manages the library through tools.

## Limitations and Biases

- **Taste is reduced to a few numbers.** The recommender sees genre, mood, energy, valence, tempo, danceability and acousticness. It ignores vocals versus instrumental, language, era and popularity, so it can't serve a listener who wants "80s" or "anything in Spanish".
- **The audio features are estimates.** `analyze_songs.py` guesses each song's energy, mood and tempo from the audio, so some songs will be mislabeled. The recommender and the AI DJ's "something sad" requests both inherit those mistakes.
- **Hand-tuned weights and a Western-leaning vocabulary.** Mood, energy and valence carry about 65% of the score, and the mood and genre tables were written by hand. A listener who cares mostly about genre gets weaker matches, and music outside the genres and moods I thought of is treated as a poor fit. Recommendations also drift toward a narrow band of similar songs, because the next pick follows the current one.
- **The agent is only as reliable as the model behind it.** Small local models sometimes guess song ids or say they did something without calling a tool. The code catches the most common cases, but it can't catch every one. The claim detector only recognizes English phrasing.
- **Limited memory and scope.** The agent keeps only the last 4 requests as context, each tool call returns at most 25 results, and it can't play an artist as a whole. It was built for a single listener, with no accounts and no multi-user support.

## Could the AI Be Misused?

- **Prompt injection through song data.** Titles, artists and album names come from file tags, so a crafted tag could contain instructions aimed at the model. Tool results go back to the model as data and the system prompt tells it to ignore attempts to change its rules, but a small model could still be fooled. The damage is limited because the agent can only call the app's own playlist, queue and search tools. It has no file, shell or network access.
- **Destructive actions.** The agent can delete or rename playlists. The prompt requires confirmation before deleting, the auto-generated mixes can't be edited, and the step cap and repeated-call refusal stop runaway loops. There is no undo, so a mistaken delete is permanent.
- **Privacy.** With Ollama everything stays on the machine. Choosing Gemini sends the chat and library details (song titles, playlist names) to a cloud service, so local models are the default and the Settings page labels Gemini as cloud.
- **Piracy and sharing.** The app plays files the user already owns and has no sharing features, but it can't verify where the files came from. It also has no login, so it should only be run on a trusted machine or network.

## What Surprised Me in Testing

- **The model's words and actions disagreed.** I expected tool calling to be straightforward, but models would reply "Added it!" without ever calling the tool, or invent a song id instead of looking it up. That is why the loop checks replies against successful tool results, and why the prompt says to act rather than announce.
- **A mock model hid real problems.** The scripted mock made the agent logic easy to test, but it never makes those mistakes, so the guard rails only proved themselves with real models.
- **Good matches had low scores.** Scores run from 0 to 1, but the 3rd to 5th recommendations often scored only around 0.5 and still matched my intuition, so a high absolute score isn't needed for a good match.

## Collaboration with AI

I built Legally Distinct-ify with an AI coding assistant, using it to draft modules and tests while I directed the features and reviewed the results.

**A suggestion I accepted: enforcing the rules in code.** The assistant proposed adding guard rails to the agent loop rather than relying on the system prompt alone: a cap of 6 model requests per message, refusing exact repeated tool calls, returning tool errors to the model as data, and a check that nudges the model when it claims an action no tool performed. I accepted it because I had already seen the model break prompt-only rules, and each guard rail is small and easy to test.

**A suggestion I changed: letting the agent edit auto-generated mixes directly.** The assistant's natural design was to give the agent one set of playlist-editing tools that worked on every playlist, mixes included, so "remove this artist from my Daily Mix" would be a single call. I scoped it down: mixes are read-only, and the agent has to call `save_mix_as_playlist` first and edit that copy. Mixes refresh through the day, so a direct edit would be silently overwritten and the agent would report a change that no longer exists. The extra step costs one more tool call, but it keeps the agent's reports true and means the AI can only change playlists that belong to the listener.
