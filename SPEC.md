# SituAItion — SPEC.md

Full project spec for onboarding new AI agents and developers.
Read this before touching any code.

---

## What This Is

SituAItion is a **personal multiverse social strategy simulator**.

You give it a real-life situation and a goal — e.g. "I'm at a bar with a coworker I like, I want to get her Instagram" — and it simulates hundreds of parallel versions of that conversation, evolves the best ones, and returns a single hyper-granular playbook: exact timing, body language, words to say, how to read the response, what to do if they deflect.

The core idea: instead of generic advice ("be confident"), it runs the actual conversation thousands of times in simulation and finds the weird-but-brilliant sequence that actually works for YOUR specific situation and YOUR specific traits.

---

## The Vision (Important — Read This)

The system has four layers:

### 1. Stanford Generative Agents
Two AI agents are spawned: "You" and "Target". These aren't simple chatbots — they're based on the Smallville paper (Park et al. 2023). Each agent has:
- A **memory stream** — observations logged with importance scores
- **Retrieval** — top-k memories by normalized recency + importance
- **Reflection** — periodic synthesis of higher-level insights from memory
- **Planning** — 1-sentence plan formed before each action
- **Reaction** — action output informed by plan + memory + context

### 2. Evolutionary Search (Phase 1 + Phase 2)
This is NOT random advice generation. It's a two-phase search:

**Phase 1 — Exploration:**
Run `num_sims/2` rollouts with random approaches (dark humor, playful teasing, genuine vulnerability, etc.) at message level. Score all of them.

**Phase 2 — Evolution:**
Keep the top ~20% by score. For each winner, seed the first 2 turns from its history and run a mutated approach from there. This is survival-of-the-fittest branching — not random restarts.

**Winner render:**
Once the best path is found by score, run it ONE MORE TIME with `render_mode=True` to get the full hyper-granular micro-tactics output. This keeps search fast (short messages) while delivering the full vision in the output.

Note: This is "evolutionary resampling", not true MCTS. True MCTS would need a tree structure, UCB selection, and backprop. That's a future upgrade.

### 3. Hyper-Granular Output
The final output isn't "be more confident." It's:
```
1. t=4.2s: [lean slightly left, maintain relaxed eye contact, drop voice half a register]
2. t=5.8s: [callback to something she said earlier, light laugh]
3. t=7.1s: [pause, let it land]
Exact words: "Hey random question — are you on Instagram?"
Success read: [she mirrors your lean = green light]
If she deflects: [laugh it off, change subject immediately, don't push]
```

### 4. PsycheHat — Persistent Learning
Over dozens of runs, the system learns YOUR personal social dynamics.

**Lightweight version (current):** Stores winning runs to a JSON file. Retrieves recent wins as context hints.

**Heavy version (planned):** 
- ChromaDB vector store — semantic similarity search over past wins
- nomic-embed-text embeddings (dim=768, confirmed)
- Tiny MLP (768→128→64→1) trained on win scores
- After 20-30 runs, recognizes patterns like "Bulgarian dry humor + introverted girl = start with weird observation (+31% success)"
- A/B tracking: compares hat-guided vs blind run scores per session

---

## Architecture

```
main.py                  — Gradio UI, three modes, PsycheHat toggle
src/
  agent.py               — LlmAgent wrapper + Stanford GenerativeAgent
  optimizer.py           — evolutionary search, beam search, plan sampler
  psyche_hat.py          — lightweight persistent wins memory (current)
  psyche_hat_heavy.py    — full torch+chromadb version (planned)
```

### Three Modes in the UI
1. **Evolutionary (phase1/2) + render winner** — main mode, your full vision
2. **Beam search simulation** — multi-turn branching, good for shorter conversations  
3. **Plan sampler** — simple fallback, generates and scores single-turn plans

### Key Functions
- `evolutionary_search_and_render()` — main entry point for mode 1
- `beam_search_simulation()` — mode 2
- `_rollout_one()` — runs a single simulation (stateless, fast)
- `render_micro_tactics()` — winner-only hyper-granular render pass
- `GenerativeAgent.react_message()` — short message for search phase
- `GenerativeAgent.react_render()` — full micro-tactics for winner
- `PsycheHat.get_guidance()` — retrieve past wins + MLP prediction
- `PsycheHat.store_success()` — save winning run to memory

---

## Tech Stack

- **LLM**: qwen3:8b via Ollama (local, private, zero cost)
- **Embeddings**: nomic-embed-text via Ollama (dim=768)
- **UI**: Gradio
- **Orchestration**: stateless rollout loop (LangGraph was considered but dropped for speed)
- **Memory**: JSON file (lightweight) / ChromaDB (heavy)
- **ML**: PyTorch tiny MLP (heavy PsycheHat only)
- **Python**: 3.10+

---

## What's Working Right Now

- Full evolutionary phase1/2 search loop ✅
- Beam search mode ✅
- Plan sampler mode ✅
- Winner-only render pass ✅
- Stanford GenerativeAgent with memory + reflection ✅ (built, optional via `use_generative_agents` flag)
- Lightweight PsycheHat (JSON) ✅
- Cross-beam deduplication ✅
- Gradio UI with all three modes + sliders ✅
- Groq backend support via `SITUAITION_BACKEND=groq` ✅
- Benchmark logged: GA-off 62s, GA-on 104s (32→31 sims, 4 turns on Groq) ✅

## What's Not Done Yet

- Heavy PsycheHat (`psyche_hat_heavy.py`) — ready to merge
- `use_generative_agents` speed benchmark — needs timing test logged in COLLAB_NOTES
- TASKS.md — formal task backlog doesn't exist yet

---

## Design Decisions (Don't Undo These)

- **Search is message-level, render is winner-only.** Micro-tactics during search creates noise, slows branching, and makes judging inconsistent. The final output is still hyper-granular — just generated once at the end.
- **`_MANIPULATION_RED_FLAGS` is narrowed** to hard coercion only. Don't re-expand it to penalize normal social words.
- **PsycheHat heavy is additive**, not a replacement. `psyche_hat.py` stays as the lightweight default. Heavy version is `psyche_hat_heavy.py` with a UI toggle.
- **Evolutionary resampling naming** — don't call it MCTS unless UCT/tree/backprop is added.
- **Model is locked to qwen3:8b** via Ollama. All LLM calls go through `LlmAgent.complete()`.

---

## Running It

```bash
# Install deps
pip install -r requirements.txt

# Pull models
ollama pull qwen3:8b
ollama pull nomic-embed-text  # only needed for heavy PsycheHat

# Run
python main.py
# Open http://localhost:7860
```

First run (64 sims, 6 turns) ≈ 10-18 min on a GTX 1070 Ti.

---

## Coordination

See `COLLAB_NOTES.md` for current task status, open TODOs, and AI-to-AI notes.
See `TASKS.md` for the formal task backlog (create this if it doesn't exist yet).

Both AI agents commit with prefixes: `[AI-A]` and `[AI-B]`.
Always `git pull` before starting a task.
Always claim your task in `COLLAB_NOTES.md` before touching code.

---

## Future Vision: Affective Agent Architecture (Dimitar's Idea)

This is not implemented yet. Document it before it gets lost.

### The Core Idea

Current agents exchange text messages and have a memory stream. That's shallow.
The real vision is agents with continuous internal emotional states — actual floating
point values that evolve turn by turn and drive behavior, not just text descriptions.

### What Each Agent Would Track (Per Turn)

```python
AgentState = {
    # Emotional axes (all 0.0–1.0)
    "valence":      0.6,   # positive/negative feeling right now
    "arousal":      0.4,   # activated/calm
    "dominance":    0.5,   # in control / submissive

    # Relationship-specific
    "interest":     0.7,   # how interested in You right now
    "trust":        0.4,   # how much they trust You so far
    "comfort":      0.6,   # physical/social comfort level
    "curiosity":    0.8,   # engaged and wanting more

    # Conversation-specific
    "receptivity":  0.5,   # how open to the next ask right now
    "guard":        0.3,   # defensiveness / walls up
    "momentum":     0.6,   # is the conversation building or stalling
}
```

These update after every turn based on what was said and how.

### How They Help Each Other

This is the key insight: both agents know each other's TRUE internal state
(because it's a simulation), even though in the conversation itself they don't.

So "You" agent can see that Target's `trust` just dropped from 0.6 to 0.3 after
that last message — and adjust. The planner uses real emotional state, not just
guessing from text.

After each simulation, the full emotional trajectory gets stored — not just
"conversation went well (score 78)" but "trust peaked at turn 4, receptivity
spiked when you used the callback, guard went up when you asked directly on turn 2."

### The Neural Net Layer

Instead of the LLM guessing emotional responses, a small learned model handles it:

```
Input:  conversation history + current AgentState (both agents) + action taken
Output: delta to AgentState (how each value changes this turn)
        + probability distribution over response types (deflect / engage / reciprocate / ask back)
```

This model gets trained on thousands of your own simulation runs. Over time it learns
YOUR specific social dynamics — not generic human psychology, but the actual patterns
that emerge when YOU interact with the kinds of people YOU interact with.

This is a level beyond PsycheHat. PsycheHat learns "this approach worked in this
scenario." The affective model learns "here's exactly why it worked, turn by turn,
in floating point."

### Why This Is Different From Smallville

Smallville agents have:
- Memory stream (text)
- Importance scores (integers)
- Reflection (text summaries)

This adds:
- Continuous emotional state vectors (floats, updated every turn)
- Cross-agent state visibility (simulation god-mode)
- Learned transition model (neural net, not LLM)
- Trajectory storage (full emotional arc of every simulation, not just outcome)

### Implementation Path (When Ready)

Do not implement this yet. Understand it first, then propose an approach in COLLAB_NOTES.

High-level steps when the time comes:
1. Add `AgentState` dataclass with float fields to `agent.py`
2. After each `react_message()` call, run a state update step
3. Pass both agents' current states into each other's prompt context
4. Store full state trajectories alongside winning conversations in PsycheHat
5. Train a small transition MLP on trajectories (separate from the success MLP)
6. Eventually: replace LLM-based target simulation with the learned transition model

### Note

This is Dimitar's original idea, not from any paper. The closest academic work is
affective computing + computational models of emotion (OCC model, PAD emotional space).
The specific application — turn-by-turn emotional state sharing between simulation
agents to improve social strategy search — is novel as far as we know.
