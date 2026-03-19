from __future__ import annotations

import os
import random
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


# ─────────────────────────────────────────────────────────────────
#  Backend selection
#
#  Set SITUAITION_BACKEND in your .env or shell:
#    SITUAITION_BACKEND=groq   → uses Groq API (fast, free tier)
#    SITUAITION_BACKEND=ollama → uses local Ollama (default)
#
#  For Groq, also set:
#    GROQ_API_KEY=your_key_here
#    SITUAITION_MODEL=llama-3.3-70b-versatile   (smarter)
#                  or llama-3.1-8b-instant       (faster, lower quality)
# ─────────────────────────────────────────────────────────────────

BACKEND = os.getenv("SITUAITION_BACKEND", "ollama").lower()


@dataclass(frozen=True)
class LlmConfig:
    model:       str   = os.getenv("SITUAITION_MODEL", "qwen3:8b")
    temperature: float = 0.85
    num_predict: int   = 450
    timeout_s:   float = 180.0
    keep_alive:  str   = "10m"   # Ollama only


class LlmAgent:
    """
    LLM wrapper supporting Ollama (local) and Groq (cloud).
    Switch via SITUAITION_BACKEND env var. Everything else is identical.
    """

    def __init__(self, *, llm: Optional[LlmConfig] = None):
        self.llm = llm or LlmConfig()
        self._client = None

    # ── Ollama backend ───────────────────────────────────────────

    def _get_ollama_client(self):
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: pip install ollama\n"
                "Then start Ollama and: ollama pull qwen3:8b"
            ) from e
        if self._client is None:
            import ollama
            timeout = float(os.getenv("SITUAITION_TIMEOUT_S", str(self.llm.timeout_s)))
            self._client = ollama.Client(timeout=timeout)
        return self._client

    def _complete_ollama(self, prompt: str, *, json_mode: bool = False) -> str:
        client = self._get_ollama_client()
        resp: dict[str, Any] = client.chat(
            model=self.llm.model,
            messages=[{"role": "user", "content": prompt}],
            format="json" if json_mode else None,
            options={
                "temperature": self.llm.temperature,
                "num_predict": self.llm.num_predict,
            },
            keep_alive=self.llm.keep_alive,
        )
        return str(resp.get("message", {}).get("content", "")).strip()

    # ── Groq backend ─────────────────────────────────────────────

    def _get_groq_client(self):
        try:
            from groq import Groq  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: pip install groq\n"
                "Then set GROQ_API_KEY in your .env file.\n"
                "Get a free key at: console.groq.com"
            ) from e
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._client

    def _complete_groq(self, prompt: str, *, json_mode: bool = False) -> str:
        client = self._get_groq_client()
        kwargs: dict[str, Any] = {
            "model":       self.llm.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": self.llm.temperature,
            "max_tokens":  self.llm.num_predict,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                resp = client.chat.completions.create(timeout=self.llm.timeout_s, **kwargs)
                return (resp.choices[0].message.content or "").strip()
            except (httpx.ReadTimeout, TimeoutError) as exc:
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(1.0)
                continue
            except Exception as exc:
                if "ReadTimeout" in str(exc) and attempt + 1 < max_attempts:
                    time.sleep(1.0)
                    continue
                raise
        raise RuntimeError("Groq completion failed after retries")

    # ── Unified interface ─────────────────────────────────────────

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        if BACKEND == "groq":
            return self._complete_groq(prompt, json_mode=json_mode)
        return self._complete_ollama(prompt, json_mode=json_mode)


# ─────────────────────────────────────────────────────────────────
#  Stanford Generative Agent  (Park et al. "Smallville")
#  observe → retrieve → reflect → plan → react
#
#  Two react modes:
#  - react_message(): short 1-2 sentence output for search phase
#  - react_render():  hyper-granular micro-tactics, winner only
# ─────────────────────────────────────────────────────────────────

@dataclass
class Memory:
    observation: str
    timestamp:   float
    importance:  int = 5  # 1–10


class GenerativeAgent:
    def __init__(
        self,
        name:          str,
        traits:        str,
        goal:          str = "",
        scenario:      str = "",
        target_traits: str = "",
        psyche_hat          = None,
        llm:           Optional[LlmAgent] = None,
    ):
        self.name          = name
        self.traits        = traits
        self.goal          = goal
        self.scenario      = scenario
        self.target_traits = target_traits
        self.psyche_hat    = psyche_hat
        self.llm           = llm or LlmAgent()
        self.memory_stream: list[Memory] = []
        self.reflections:   list[str]   = []
        self.clock:         float       = 0.0

    def observe(self, obs: str, importance: int = 5):
        self.memory_stream.append(
            Memory(observation=obs, timestamp=time.time(), importance=importance)
        )

    def retrieve(self, k: int = 5) -> str:
        """Top-k by normalized recency + importance (both 0-1)."""
        recent = self.memory_stream[-20:]
        if not recent:
            return ""
        now     = time.time()
        oldest  = min(m.timestamp for m in recent)
        t_range = max(now - oldest, 1.0)

        def score(m: Memory) -> float:
            return 0.3 * (m.timestamp - oldest) / t_range + 0.7 * (m.importance - 1) / 9.0

        scored = sorted(recent, key=score, reverse=True)
        return textwrap.shorten("\n".join(m.observation for m in scored[:k]), width=800)

    def reflect(self):
        if len(self.memory_stream) < 4:
            return
        insight = self.llm.complete(
            f"You are {self.name}. Write 2-3 brief insights about the social dynamic "
            f"based on:\n{self.retrieve()}"
        )
        if insight:
            self.reflections.append(insight)

    def plan(self, situation: str) -> str:
        ref_str = "\n".join(self.reflections[-2:]) if self.reflections else "None yet."
        return self.llm.complete(
            f"You are {self.name} ({self.traits}). Goal: {self.goal}.\n"
            f"Memory: {self.retrieve()}\nReflections: {ref_str}\n"
            f"Situation: {situation}\nYour plan for the next moment (1 sentence):"
        )

    def _hat_line(self) -> str:
        if not self.psyche_hat:
            return ""
        try:
            g = self.psyche_hat.get_guidance(
                scenario=self.scenario,
                goal=self.goal,
                you_traits=self.traits,
                target_traits=self.target_traits,
            )
            return f"\nPsycheHat: {g['tip']}"
        except Exception:
            return ""

    def react_message(self, situation: str, approach: str = "") -> str:
        """Short 1-2 sentence output for search phase."""
        return self.llm.complete(
            f"You are {self.name} ({self.traits}). Goal: {self.goal}.\n"
            f"Plan: {self.plan(situation)}\nMemory: {self.retrieve()}\n"
            f"Approach: {approach}{self._hat_line()}\nSituation: {situation}\n\n"
            f"Write your next message in 1-2 natural sentences. Human and specific."
        )

    def react_render(self, situation: str, approach: str = "") -> str:
        """Hyper-granular micro-tactics. Called ONCE on the winning branch only."""
        t = self.clock
        resp = self.llm.complete(
            f"You are {self.name} ({self.traits}). Goal: {self.goal}.\n"
            f"Plan: {self.plan(situation)}\nMemory: {self.retrieve()}\n"
            f"Approach: {approach}{self._hat_line()}\nSituation: {situation}\n\n"
            f"Output HYPER-GRANULAR micro-tactics. Specific — not generic advice.\n"
            f"FORMAT:\n"
            f"1. t={t + random.uniform(0.5, 1.5):.1f}s: [micro-action + body language + tone]\n"
            f"2. t=XX.Xs: [next action]\n"
            f"3. t=XX.Xs: [next action]\n"
            f'Exact words: "..."\n'
            f"Success read: [how to tell if it's landing]\n"
            f"If they deflect: [adjust and do this]\n"
        )
        self.clock += random.uniform(1.4, 3.8)
        return resp
