# SituAItion — SPEC.md

Full project spec for onboarding new AI agents and developers.
Read this before touching any code.

---

## What This Is

SituAItion is a **personal multiverse social strategy simulator**.

You describe any real-life situation and a goal. The system simulates hundreds of parallel versions of how it could play out, evolves the best ones, and returns a single hyper-granular playbook: exact timing, body language, words, how to read signals, what to do if things go sideways.

The core idea: instead of generic advice ("be confident"), it runs the actual situation thousands of times in simulation and finds the weird-but-brilliant sequence that actually works — including non-obvious emergent tactics that no human would explicitly think of.

### What "situation" actually means

NOT just romantic scenarios. The input is any interpersonal situation with a goal:

- "I'm outside a café. She's reading alone, earbuds in, body language closed. 3 other people nearby. From what I can see she likes coffee and probably reads fiction. Goal: start a conversation without being weird."
- "I'm in a meeting. My manager keeps shooting down my ideas before I finish. Goal: get my proposal actually heard."
- "My friend group is planning something I don't want to do. Goal: redirect without being the difficult one."
- "I'm negotiating salary. They just gave me a number. Goal: counter without damaging the relationship."

The system treats ALL of these the same way: observe the state, model the agents involved, simulate thousands of paths, evolve the winners, output the playbook.

### What "novel tactics" means

The evolutionary search doesn't just generate scripted dialogue. Over thousands of simulations it should surface emergent non-obvious micro-moves — things like:

- A well-timed pause before responding that shifts perceived confidence
- Looking away at a specific moment to reduce pressure
- Referencing something environmental rather than the person directly
- A specific laugh cadence that signals you're not trying too hard

These are the moves that don't appear in any social advice article because they're too subtle and context-specific. They only emerge from actually simulating the interaction at scale.

**Currently:** the LLM converges on safe scripted social dialogue. The evolutionary search finds "best script" not "novel emergent tactic."
**Future (PAD + NN):** the transition model learns which micro-state changes produce which outcomes, enabling genuinely non-obvious recommendations.

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
- Render pass token budget bumped to 1000 so micro-tactics stop truncating ✅
- Judge incentivizes non-generic tactics via novelty reward ✅
- PAD AgentState foundation (Steps 1-3: state tracking + cross-agent visibility) ✅

## What's Not Done Yet

- Situation input redesign — UI currently assumes conversation; should accept observed state (environment, body language, inferred interests)
- PAD + AgentState implementation — see Future Vision section
- Transition NN training — needs trajectory data first
- TASKS.md — formal task backlog doesn't exist yet

---

## Design Decisions (Don't Undo These)

- **Search is message-level, render is winner-only.** Micro-tactics during search creates noise, slows branching, and makes judging inconsistent. The final output is still hyper-granular — just generated once at the end.
- **`_MANIPULATION_RED_FLAGS` is narrowed** to hard coercion only. Don't re-expand it to penalize normal social words.
- **PsycheHat heavy is additive**, not a replacement. `psyche_hat.py` stays as the lightweight default. Heavy version is `psyche_hat_heavy.py` with a UI toggle.
- **Evolutionary resampling naming** — don't call it MCTS unless UCT/tree/backprop is added.
- **Backend is switchable** via `SITUAITION_BACKEND` env var (ollama or groq). Default is now Groq. All LLM calls go through `LlmAgent.complete()` regardless of backend.

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

Agent emotional state is anchored to the **PAD model** (Mehrabian & Russell, 1974)
— a validated, universal three-dimensional emotional space used in affective computing.
Using PAD instead of arbitrary fields keeps the system academically grounded and
universal across users, not personal/idiosyncratic.

```python
AgentState = {
    # Universal PAD layer (Mehrabian & Russell 1974) — academically grounded
    "pleasure":     0.6,   # -1 to 1 — positive/negative valence
    "arousal":      0.4,   # -1 to 1 — activated/calm
    "dominance":    0.5,   # -1 to 1 — in control / submissive

    # Social layer — PAD can't capture these, but they matter for goal-directed sims
    "trust":        0.4,   # 0 to 1 — has she opened up yet
    "interest":     0.7,   # 0 to 1 — attracted/engaged vs indifferent
    "receptivity":  0.5,   # 0 to 1 — readiness for the specific ask right now
    "momentum":     0.6,   # 0 to 1 — is the conversation building or stalling
}
```

7 values total. PAD handles universal emotional foundation. The social layer handles
what PAD can't — trust, interest, and readiness for the ask are not derivable from
pleasure/arousal/dominance alone.

PAD combinations that matter:
- High P + High A + Low D = excited vulnerability (ideal target state for romantic ask)
- High P + Low A + High D = relaxed confidence (ideal "You" state)
- Low P + High A + Low D = anxious/defensive (target feeling pressured — back off)
- Low P + Low A + Low D = disengaged/bored (conversation dying)

All values update as weighted time series each turn, influenced by the last ~12
events weighted by recency — consistent with OCC+PAD literature on mood evolution.

### How the Float Values Actually Influence the LLM

Raw floats mean nothing to an LLM — `trust=0.1` is like `print(GTA5)`.
The values must be **verbalized** into natural language before being injected
into the prompt. A `pad_to_text()` function converts state to human-readable
conditioning:

```python
def state_to_text(state: AgentState) -> str:
    lines = []
    if state["pleasure"] < -0.3:
        lines.append("feeling negative and guarded")
    elif state["pleasure"] > 0.3:
        lines.append("feeling positive and warm")
    if state["trust"] < 0.3:
        lines.append("low trust — hasn't opened up yet")
    elif state["trust"] > 0.7:
        lines.append("high trust — comfortable and open")
    if state["receptivity"] > 0.7:
        lines.append("receptive — good moment to ask")
    elif state["receptivity"] < 0.3:
        lines.append("not receptive right now — back off")
    if state["momentum"] < 0.3:
        lines.append("conversation stalling")
    return ", ".join(lines)
```

This is like positive/negative conditioning in text-to-image generation —
the float values are the conditioning signal, verbalization makes them
meaningful to the model, and the LLM adjusts its output accordingly.

### The Full Pipeline Per Turn

```
Float state values
      ↓
state_to_text() → natural language conditioning
      ↓
Injected into both agents' prompts as context
      ↓
LLM generates action (message/response)
      ↓
LLM also outputs delta JSON: {"pleasure": +0.1, "trust": +0.2, ...}
      ↓
Float values updated
      ↓
(state, action, delta) tuple stored as trajectory data
      ↓
NN trains on tuples offline:
  learns: action X in state Y → produces delta Z
  eventually replaces LLM delta output entirely
```

### How They Help Each Other

Both agents know each other's TRUE internal state (because it's a simulation),
even though in the conversation itself they don't.

So "You" agent sees Target's verbalized state: "feeling guarded, low trust, not
receptive" — and adjusts its next move accordingly. The planner acts on real
emotional state, not just guessing from text alone.

After each simulation, full state trajectories get stored — not just "score 78"
but the complete arc: "trust peaked at turn 4, receptivity spiked at the callback,
momentum dropped when you asked too early on turn 2."

### The Neural Net Layer

Instead of the LLM guessing emotional responses, a small learned model handles it.
The PAD values serve directly as weights in the transition NN:

```
Input:  PAD state of both agents (6 values) + action embedding
Output: delta PAD (how each value shifts this turn)
        + response type probability distribution (deflect / engage / reciprocate / ask back)
```

Because PAD is a universal validated space, this model learns general human emotional
response patterns — not personal/idiosyncratic dynamics. PsycheHat adds the personal
layer on top optionally.

This is a level beyond PsycheHat. PsycheHat learns "this approach worked in this
scenario." The affective model learns "here's exactly why it worked, turn by turn,
in floating point — and generalizes that to anyone."

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

Steps 1-2 are done. Steps 3-6 are next.

1. ✅ Add `AgentState` dataclass with float fields to `agent.py` — done by AI-B
2. ✅ After each `react_message()` call, run a state update step — done by AI-B
3. ✅ Pass both agents' current states into each other's prompt context — "You" now sees Target's verbalized PAD state and gets a receptivity guardrail
4. Store full state trajectories alongside winning conversations in PsycheHat
5. Train a small transition MLP on trajectories (separate from the success MLP)
6. Eventually: replace LLM-based target simulation with the learned transition model

### Known Limitation: State Update Heuristic (fix before NN training)

Current implementation uses `update_from_text()` — a keyword heuristic that guesses
PAD delta values from words in the LLM response. This is intentionally lightweight
for the foundation but will produce noisy/wrong state updates frequently.

Example of the problem: if the LLM response contains "I feel nervous" the heuristic
might correctly drop pleasure, but if it says "I laughed nervously" it might miss
the social signal entirely.

**Before training the NN on trajectory data, this must be replaced with:**
```
LLM outputs delta JSON explicitly:
{"pleasure": -0.1, "arousal": +0.2, "dominance": -0.1, "trust": +0.0, ...}
```
Add a structured output step after each `react_message()` call where the LLM
is asked: "Given this exchange, how did Target's emotional state shift? Output
only JSON with keys: pleasure, arousal, dominance, trust, interest, receptivity, momentum."

Training the NN on heuristic-generated trajectories would teach it noise, not signal.
Fix this before Step 5.

### Note

This is Dimitar's original idea, not from any paper. The closest academic work is
affective computing + computational models of emotion (OCC model, PAD emotional space).
The specific application — turn-by-turn emotional state sharing between simulation
agents to improve social strategy search — is novel as far as we know.
