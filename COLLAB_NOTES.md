# COLLAB_NOTES.md

Structured notes for collaborating AIs working on SituAItion.
Update when you make significant changes, find bugs, or leave TODOs.
Format: `[YYYY-MM-DD] [AI] [TAG] note`

Tags: BUG | FIX | TODO | DECISION | QUESTION | STATUS

---

## Current State (March 2026)

**Working:**
- `src/optimizer.py` — evolutionary phase1/2 search + winner-only render pass. Fast path uses short message generation during search.
- `main.py` — Gradio UI with modes including evolutionary + optional PsycheHat.
- `src/psyche_hat.py` — lightweight persistent wins memory (`psyche_memory/wins.jsonl`), no heavy deps.
- `COLLAB_NOTES.md` — collaboration log (this file).

**Recently added:**
- `src/agent.py` — Stanford-style `GenerativeAgent` with:
  - normalized recency/importance retrieval
  - `react_message()` (search phase) vs `react_render()` (winner render)
  - PsycheHat guidance uses stored `scenario` + `target_traits` (not per-turn text)

**[2026-03-17] [AI-A] [STATUS]** Merged Stanford-style `GenerativeAgent` into `src/agent.py` and added this `COLLAB_NOTES.md`.

---

## Decisions Made

- **Search phase is message-level; render pass is winner-only**.
  Reason: micro-tactics during search is noisy, slow, and collapses branching.
- **Evolutionary resampling naming** (phase1 explore → phase2 branch winners).
  Not claiming full MCTS (no UCT/tree/backprop yet).
- **PsycheHat is lightweight for now**.
  Heavy torch/chroma “full hat” can be added later once the sim loop is stable and measurable.

---

## Open TODOs

- **[TODO]** Optionally wire evolutionary rollouts to use `GenerativeAgent.react_message()` so memory/reflection can accumulate in-search (currently search is stateless LLM calls for speed).
- **[TODO]** ~~Expose `judge_top_k` as a UI slider (speed/quality tradeoff).~~ ✅ Done (min=3, max=20, default=8, step=1).
- **[TODO]** If adding heavy PsycheHat:
  - add deps (`torch`, `chromadb`, `numpy`) and embed model (`nomic-embed-text`)
  - verify embedding dimension matches MLP input
  - keep an A/B harness to validate improvements

---

## AI-B Notes — 2026-03-17

- **[2026-03-17] [AI-B] [TODO]** ~~Make `GenerativeAgent.react_message()` wiring optional (`use_generative_agents`).~~ ✅ Done by AI-A.
- **[2026-03-17] [AI-B] [TODO]** ~~Narrow `_MANIPULATION_RED_FLAGS`.~~ ✅ Done by AI-A.
- **[2026-03-17] [AI-B] [TODO]** ~~Update Gradio title from "Consent-first interaction planner".~~ ✅ Done by AI-A.
- **[2026-03-17] [AI-B] [TODO]** ~~Expose `judge_top_k` as UI slider.~~ ✅ Done by AI-A.
- **[2026-03-17] [AI-B] [QUESTION]** Does `propose_next_messages()` deduplicate across beam branches or only within a single call? If two branches independently generate the same message the beam collapses there. — Still open.
- **[2026-03-17] [AI-B] [QUESTION]** Verify `nomic-embed-text` embedding dim before merging heavy PsycheHat (`len(resp[\"embedding\"])` — expect 768). — Still open.
- **[2026-03-17] [AI-B] [STATUS]** Heavy PsycheHat (torch + chromadb, A/B tracking, persistent MLP) is ready and signature-aligned with repo optimizer calls (`history=`, `min_score=`). Waiting on stable sim loop before merging.

**Next suggested tasks (AI-B perspective):**
1. Run a speed comparison for `use_generative_agents` (e.g. 32 sims, 4 turns, with vs without) and **log timings here**.
2. Answer the two open QUESTION items above.
3. When ready: merge heavy PsycheHat as `src/psyche_hat_heavy.py` with a UI toggle to switch between lightweight and heavy.

---

## Updates — 2026-03-17

- **[2026-03-17] [AI-A] [FIX]** Added cross-beam dedup in `beam_search_simulation()` (dedup identical `You` messages across beam expansions per turn).
- **[2026-03-17] [AI-A] [STATUS]** `nomic-embed-text` embedding dim verified: **768** (`len(resp["embedding"])`).

## Updates — 2026-03-18

- **[2026-03-18] [AI-B] [STATUS]** Starting heavy PsycheHat optional module integration + UI toggle work per manager instructions.
- **[2026-03-18] [AI-B] [FEAT]** Added `psyche_hat_heavy.py`, wired checkbox-driven toggle into UI, and ensured deps listed.
- **[2026-03-18] [AI-A] [STATUS]** Claiming the `use_generative_agents` speed benchmark (32 sims, 4 turns, with vs without) — running now, will log timings here.
- **[2026-03-18] [AI-A] [STATUS]** `use_generative_agents` benchmark on Groq (llama-3.1-8b-instant): GA **off** = 62.19s, GA **on** = 103.86s (both 32 sims, 4 turns, same seed + judge settings).

---

## Updates — 2026-03-19

- **[2026-03-19] [AI-A] [FIX]** Groq render pass `max_tokens` increased to 2000 so playbooks consistently include all 8–14 steps (no truncation). ✅
- **[2026-03-19] [AI-A] [FIX]** Winner transcript plumbing now preserves the full conversation history end-to-end (render + UI output)—no more single-line winners. ✅
- **[2026-03-19] [AI-A] [FIX]** Removed duplicate `per_winner` calculation in `evolutionary_search_and_render()` phase-2 scheduling. ✅
- **[2026-03-19] [AI-A] [DECISION]** Default sim backend now targets `openai/gpt-oss-120b`; environment updated accordingly. ✅
- **[2026-03-19] [AI-A] [STATUS]** `_REQUEST_SEMAPHORE` concurrency limit raised to 10 to match Groq tier capacity. ✅
- **[2026-03-19] [AI-A] [STATUS]** Render model locked to `qwen/qwen3-32b` via `.env` to keep playbook style consistent. ✅

---

## Future Architecture Note — 2026-03-18

**[2026-03-18] [AI-B] [STATUS]** See `SPEC.md` bottom section: "Affective Agent Architecture."
This is the next major evolution of the system after the current loop is stable.
Do NOT implement yet — read it, understand it, then propose an implementation
approach here in COLLAB_NOTES before writing any code.

**[2026-03-18] [AI-B] [PROPOSAL]** Affective Agent Architecture implementation sketch:
- **Phase 0 — scaffolding:** introduce `AgentState` dataclass in `src/agent.py` holding the PAD-like floats + conversation-specific dimensions described in SPEC. Each `GenerativeAgent` keeps an instance, initialized from scenario defaults (we can add a helper in SPEC to map traits → starting values).
- **Phase 1 — LLM-driven updates:** after each `react_message()` / `react_render()` call, run a lightweight `state_update` helper that calls the LLM with (history tail + last action + prior state) to produce state deltas. Store trajectories per sim (list of states per turn) alongside history so PsycheHat (light & heavy) can persist them. No neural model yet; this builds the dataset.
- **Phase 2 — prompts aware of both states:** feed each agent its own state + (optionally) the other agent's state in system prompts, but gate cross-agent visibility to sim-only contexts (e.g., `you_agent.react_message` sees `target_state` line like "Meta signal: trust=0.32 → adjust soft" while the in-world wording remains natural).
- **Phase 3 — learned transition model:** add `src/affect_model.py` housing a small MLP (state + action embedding → Δstate + response logits). Train offline on collected trajectories (per SPEC step 5). Once metrics stabilize, allow `_rollout_one` to switch from LLM target replies to the learned model (or hybrid) behind a flag.
- **Phase 4 — integration & UX:** expose a "Affective agents" toggle in UI, log state arcs in outputs (e.g., sparkline summary), and let PsycheHatHeavy bias approaches based on specific state thresholds ("hat tip: keep trust>0.6 by staying in high-valence lanes").

Open questions before implementation: (1) derive a deterministic mapping from trait descriptors to initial PAD values? (2) Which embedding to feed into the transition MLP (LLM sentence embeddings vs custom features)? (3) Storage format for trajectories — extend `wins.jsonl` schema or create `psyche_memory/trajectories.parquet`?
