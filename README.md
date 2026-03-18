# SituAItion — personal multiverse social strategy simulator

SituAItion runs dozens of believable parallel conversations for a single real-world scenario, evolves the best emergent branches, then renders a hyper-granular playbook (timing, body language, exact phrasing, fallbacks). Instead of vague advice, it finds the weird-but-effective tactics that fit **your** goal and traits.

---

## Features

- **Evolutionary search (phase1 explore → phase2 branch winners)** with winner-only micro-tactics rendering.
- **Beam search sandbox** for branching simulations without the evolutionary loop.
- **Plan sampler** as a fast fallback when you just need single-turn ideas.
- **PsycheHat memory**: lightweight JSON wins log today, optional heavy module (torch + Chroma) coming online.
- **Optional Stanford Generative Agents** (`use_generative_agents` flag) so rollouts can accumulate memories/reflections when you need coherence more than raw speed.
- **Gradio UI** with sliders/toggles for sims, turns, judge top-K, hat backend, A/B testing, and Generative Agents.
- **Benchmark harness** (`scripts/use_generative_agents_benchmark.py`) for comparing GA on/off performance (32 sims, 4 turns by default).

---

## Architecture

```
main.py                  — Gradio UI entry point
src/
  optimizer.py           — evolutionary loop, beam search, plan sampler
  agent.py               — LlmAgent wrapper + Stanford GenerativeAgent
  psyche_hat.py          — lightweight persistent wins memory (JSONL)
  psyche_hat_heavy.py    — heavy torch + chromadb hat (in progress)
psyche_memory/           — stored wins
scripts/use_generative_agents_benchmark.py — GA speed test harness
```

Key design decisions:

- Search mode stays **message-level** for speed/noise control; the winner gets a single render pass for micro-tactics.
- `_MANIPULATION_RED_FLAGS` only covers hard coercion; normal social language is allowed.
- PsycheHat heavy is additive. Lightweight hat remains default and costs zero extra deps.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `qwen3:8b` pulled (`ollama pull qwen3:8b`).
- `pip install -r requirements.txt`
- Heavy PsycheHat (optional) also needs `torch`, `chromadb`, `numpy`, and `ollama pull nomic-embed-text`.

---

## Quick start

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
# optional for heavy hat
ollama pull nomic-embed-text

python main.py
# open http://localhost:7860
```

UI tips:

1. Pick a mode (evolutionary by default) and fill in scenario/goal/traits.
2. Enable PsycheHat (Light/Heavy) or Generative Agents only when you need extra context—they slow sims down.
3. For evolutionary mode, tweak `Evolutionary sims`, `Turns per sim`, and `Judge top-K` for the speed/quality tradeoff you need.

---

## Benchmarking `use_generative_agents`

The repo includes a reproducible benchmark comparing rollouts with/without Generative Agents:

```bash
python scripts/use_generative_agents_benchmark.py
```

It runs evolutionary mode twice (32 sims, 4 turns) and prints per-run timing plus metadata (`phase1`, `total_sims`, `turns`). Make sure Ollama is running before executing it.

Log the resulting numbers in `COLLAB_NOTES.md` so collaborators can see the speed delta.

---

## Collaboration

- Always `git pull` before starting work and note your task in `COLLAB_NOTES.md`.
- Commit with `[AI-A]` or `[AI-B]` prefixes depending on which assistant performed the change.
- Keep PsycheHat heavy/optional work isolated—the lightweight hat stays the default path.

---

## License

See [LICENSE](LICENSE) for details.
