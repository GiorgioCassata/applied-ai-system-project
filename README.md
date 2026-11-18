 # Explicitly name your original project (from Modules 1-3) and provide a 2-3 sentence summary of its original goals and capabilities.
 The original project was the Music Recommender Simulation. The purpose of the recommender was to identify and classify various properties of songs to be used in song recommendation. This was to mimic the type of recommendation systems used in major music streaming applications that caters to what individual users like.

# Title and Summary: What your project does and why it matters.
Legally Distinct-ify is a music streaming app for local music libraries. It provides a sleek interface that offers a sensible auto-play and AI playlist generation. This allows individuals with a local library to enjoy a modern listening experience without paying any subscriptions. 

# Architecture Overview: A short explanation of your system diagram.
At a base level, the user interacts solely with the streamlit server. The different views, panels, searchbars, and caches use supporting modules to perform critical operations such as storing a few files of interest temporarily on the users computer. album_art.py finds files and cover art. search.py matches text to song titles with some helpful filtering and matching. user_profile.py handles the user profile file and updates. play_queue.py is a class that handles the progression of playing songs and any modifying actions. playlists.py handles everything from the auto-generated daily mixes to the user playlist creation. recommender.py is the heart of building mixes and continuing queues by scoring songs based off of their properties and using songs' scores to rank and recommend following songs. 


# Setup Instructions: Step-by-step directions to run your code.
1. Install the dependencies:
   `pip install -r requirements.txt`

2. Populate songs.csv with your library and build the album-art thumbnails:
   `python scripts/populate_songs.py /Volumes/{Your path here}`
   It saves its progress as it goes, so it can be stopped (Ctrl-C) and re-run.

3. Estimate each new song's energy, mood and tempo, and update songs.csv:
   `python scripts/analyze_songs.py`

4. (Optional) Link a language model for the AI DJ. The rest of the app works without one.
   - **Ollama (local, default):** install [Ollama](https://ollama.com) and pull a model that supports tool calling, e.g. `ollama pull gemma4:latest`. More models can be added to `OLLAMA_MODELS` in `src/llm_client.py`.
   - **Gemini (cloud):** create a `.env` file containing `GEMINI_API_KEY=your-key`.
   - **Mock:** offline scripted replies, for testing.

   You can choose the model later on the app's Settings page.

5. Start the app:
   `streamlit run src/app.py`

# Sample Interactions: Include at least 2-3 examples of inputs and the resulting AI outputs to demonstrate the system is functional.

- Input: "Play something sad."
- Tools called: `find_songs_by_feel` (low valence), then `play_song`.
- Output: "I found a few sad songs. I'm starting "Inside" by Sting now."

- Input: "Make a playlist called Late Night with some chill lofi and play it."
- Tools called: `find_songs_by_feel`, `create_playlist`, `read_playlist`, `play_playlist`.
- Output: "I'm sorry, but "lofi" isn't a genre in the app. I can search for songs that are "chill" or in a related genre like "easy listening" or "new age." Which would you prefer?"

- Input: "Can you make me a playlist with 12 songs from The beatles, lady gaga, and david bowie that will make me happy?"
- Tools called: `find_song` (once each for The Beatles, Lady Gaga and David Bowie), `read_song` (to check each candidate's valence, picking the upbeat ones), `create_playlist` (the 12 chosen song ids), `read_playlist` (to confirm the result).
- Output: "I made a playlist called "Happy Pop Mix" for you. It has 12 songs from The Beatles, Lady Gaga, and David Bowie."

# Design Decisions: Why you built it this way, and what trade-offs you made.
- **Local-first.** Music stays on the user's own drive and the app runs on their machine. Ollama is the default model so chat works with no cloud account and no data leaving the computer. Trade-off: small local models are slower and less reliable at tool calling than cloud ones.
- **Guard rails in code, not just the prompt.** At most 6 model requests per message, exact repeated tool calls are refused, replies that claim an action with no successful tool result are caught, and a failed model request gives a fixed friendly reply. Trade-off: more code to maintain, but it holds even when the model ignores its instructions.
- **Tools reuse the app's own modules.** The agent calls the same playlist, queue and search code as the UI, so chat can't do anything the interface can't, and there is one source of truth.
- **Agent runs in a background thread.** Streamlit re-runs the page on every click, which would cut an inline agent off half way.
- **Hand-tuned recommender weights.** Mood, energy and valence count the most because songs that agree on those flow well together. Simple and explainable, but chosen by hand rather than learned from listeners.
- **Plain JSON and CSV storage.** Easy to inspect and back up, but it wouldn't scale past a single user.

# Testing Summary: What worked, what didn't, and what you learned.
The project has about 245 automated unit tests (`pytest`) covering the recommender, play queue, playlists, search, user profile, agent settings, every agent tool, and the agent loop. The agent tests use a scripted mock model, so they are deterministic and need no network.

**What worked:** testing the agent against a mock model made its guard rails (step limit, repeated-call refusal, unverified-claim nudge, graceful failure) easy to verify. Recommender profiles built around individual songs confirmed the expected songs ranked on top.

**What didn't:** real models were less predictable than the mock. Small local models sometimes announced an action without calling the tool, or guessed ids. Recommender scores for the 3rd to 5th picks were often only around 0.5, so a high absolute score isn't needed for a reasonable match.

**What I learned:** prompts alone are not a reliable safety net. Anything that must hold, like never claiming an action that wasn't performed, should be enforced in code and tested.

# Reflection: A brief note on what this project taught you about AI and problem-solving. Your graded responsible-AI 

Building Legally Distinct-ify showed me that an AI feature is only as trustworthy as the system around it. Wiring up the model was the easy part. Most of the work was constraining it: giving it small, well-described tools, returning errors as data so it can recover, and checking in code that what it says matches what actually happened. I also learned to treat the model as a setting rather than a dependency, since a local model and a cloud model fail in different ways.

# reflection — how you collaborated with AI, one suggestion you accepted and one you rejected or revised, and your 

I relied heavily on Claude for this project. I developed iteratively, acting as the end user providing feedback, and the manager in directing fixes and outlining goals. However, many changes I accepted with little review, instead relying on tests. Originally Claude was doing a lot of testing and checks, even running the app in a simulated environment at one point. This was extremely slow and was overkill in terms of the tokens it was burning. So I rejected this methodology and the supporting test file changes. 

# system's limitations — goes in model_card.md (see Step 5), not here. Reflection content placed only in the README does not earn the reflection points.



