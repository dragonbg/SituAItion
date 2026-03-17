# SituAItion
Been there done that - Personal multiverse social strategy simulator – runs thousands of simulations to find novel micro-tactics for real-life goals

# SituAItion

**The AI that has already lived every possible version of this conversation.**

Personal "multiverse god" social strategy oracle:  
Input any real-life interpersonal situation + goal → it spawns believable agents (You + Target), runs thousands of parallel short simulations via Monte Carlo rollouts, evolves the best emergent paths, and outputs hyper-granular, novel micro-tactics with precise timing, body language, exact phrases, and success estimates.

Often surfaces weird-but-effective emergent actions (e.g. "light kneecap touch at t=4.7s + specific posture shift + laugh cadence") that beat generic advice.

## Core Goal & Use Cases
- Make X do Y (without clichés)
- Increase romantic interest / rapport in subtle ways
- Negotiation, sales pitches, networking, conflict resolution
- Ethical social skills simulator (with guardian veto planned)

## Architecture
Inspired by Stanford Generative Agents (Smallville) paper + modern optimizations:

- **Agents**: Custom GenerativeAgent class (Memory Stream, Reflection, Reaction modules)
- **Orchestration**: LangGraph (stateful graphs, cycles)
- **Optimization**: Monte Carlo rollouts (thousands of parallel short sims) → pick/evolve best path (MCTS/genetic planned next)
- **LLM**: Local via Ollama + LiteLLM (qwen2.5:14b primary, 8b fallback)
- **UI/Access**: Gradio web UI → future Telegram bot / PWA for phone use

## Current Status (MVP)
- Runnable prototype with basic agent sim + Monte Carlo scoring
- Gradio interface for quick testing
- Local-only, zero-cost, private

## Quick Start

1. Install Ollama & pull model:
   ```bash
   ollama pull qwen2.5:14b
