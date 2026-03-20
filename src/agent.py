from __future__ import annotations

import json
import os
import random
import textwrap
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

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


def _safe_json_loads(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class LlmConfig:
    model:             str   = os.getenv("SITUAITION_MODEL", "openai/gpt-oss-120b")
    temperature:       float = 0.85
    num_predict:       int   = 450
    timeout_s:         float = 180.0
    keep_alive:        str   = "10m"   # Ollama only


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
            "service_tier": "auto",
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

STATE_FIELDS: tuple[str, ...] = (
    "pleasure",
    "arousal",
    "dominance",
    "trust",
    "interest",
    "receptivity",
    "momentum",
)


@dataclass
class AgentState:
    """PAD + social layer values in [-1, 1] / [0, 1]."""

    pleasure:    float = 0.05  # -1..1
    arousal:     float = 0.05  # -1..1
    dominance:   float = 0.05  # -1..1
    trust:       float = 0.4   # 0..1
    interest:    float = 0.45  # 0..1
    receptivity: float = 0.35  # 0..1
    momentum:    float = 0.45  # 0..1

    def _clamp(self) -> None:
        def _clip(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        self.pleasure = _clip(self.pleasure, -1.0, 1.0)
        self.arousal = _clip(self.arousal, -1.0, 1.0)
        self.dominance = _clip(self.dominance, -1.0, 1.0)
        self.trust = _clip(self.trust, 0.0, 1.0)
        self.interest = _clip(self.interest, 0.0, 1.0)
        self.receptivity = _clip(self.receptivity, 0.0, 1.0)
        self.momentum = _clip(self.momentum, 0.0, 1.0)

    def apply_delta(self, delta: dict[str, float]) -> None:
        for field in STATE_FIELDS:
            val = delta.get(field)
            if isinstance(val, (int, float)):
                setattr(self, field, getattr(self, field) + float(val))
        self._clamp()

    def to_text(self) -> str:
        bits: list[str] = []
        if self.pleasure >= 0.35:
            bits.append("feeling warm")
        elif self.pleasure <= -0.4:
            bits.append("feeling cold/guarded")
        else:
            bits.append("neutral mood")

        if self.arousal >= 0.35:
            bits.append("energized")
        elif self.arousal <= -0.35:
            bits.append("calm/low energy")

        if self.trust >= 0.7:
            bits.append("high trust")
        elif self.trust <= 0.3:
            bits.append("low trust")

        if self.receptivity >= 0.7:
            bits.append("receptive to the ask")
        elif self.receptivity <= 0.2:
            bits.append("not ready for the ask")

        if self.momentum <= 0.35:
            bits.append("conversation stalling")
        elif self.momentum >= 0.65:
            bits.append("conversation building")

        return ", ".join(bits) or "neutral baseline"

    def debug_summary(self) -> str:
        data = {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "trust": self.trust,
            "interest": self.interest,
            "receptivity": self.receptivity,
            "momentum": self.momentum,
        }
        return " | ".join(f"{k}={v:+.2f}" for k, v in data.items())



@dataclass
class Memory:
    observation: str
    timestamp:   float
    importance:  int = 5  # 1–10


class GenerativeAgent:
    _STATE_FIELDS: ClassVar[tuple[str, ...]] = STATE_FIELDS

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
        base_cfg = self.llm.llm
        self._state_delta_llm = LlmAgent(
            llm=LlmConfig(
                model="llama-3.3-70b-versatile",
                temperature=base_cfg.temperature,
                num_predict=150,
                timeout_s=base_cfg.timeout_s,
                keep_alive=base_cfg.keep_alive,
            )
        )
        self.memory_stream: list[Memory] = []
        self.reflections:   list[str]   = []
        self.clock:         float       = 0.0
        self.state          = self._initial_state()

    def _initial_state(self) -> AgentState:
        state = AgentState()
        traits = (self.traits or "").lower()
        if "confident" in traits or "assertive" in traits:
            state.dominance += 0.2
            state.trust += 0.05
        if "introvert" in traits or "reserved" in traits:
            state.arousal -= 0.15
            state.receptivity -= 0.05
        if "playful" in traits or "warm" in traits:
            state.pleasure += 0.2
        state._clamp()
        return state

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

    def react_message(
        self,
        situation: str,
        approach: str = "",
        *,
        partner_state_text: str | None = None,
        partner_receptivity: float | None = None,
        environment: str = "",
    ) -> str:
        """Short 1-2 sentence output for search phase."""
        plan_line = self.plan(situation)
        partner_line = ""
        if partner_state_text:
            partner_line = f"Target's current state: {partner_state_text.strip()}\n"
            if partner_receptivity is not None and partner_receptivity < 0.4:
                partner_line += "Target is not ready for a direct ask yet — build more rapport first.\n"
        env_line = f"Observed environment: {environment.strip()}\n" if environment.strip() else ""
        resp = self.llm.complete(
            f"You are {self.name} ({self.traits}). Goal: {self.goal}.\n"
            f"Internal state: {self.state.to_text()}\n"
            f"{partner_line}"
            f"{env_line}"
            f"Plan: {plan_line}\nMemory: {self.retrieve()}\n"
            f"Approach: {approach}{self._hat_line()}\nSituation: {situation}\n\n"
            f"Write your next message in 1-2 natural sentences. Human and specific."
        )
        self._update_state_with_structured_delta(
            situation=situation,
            latest_message=resp,
            partner_state_text=partner_state_text,
            environment=environment,
        )
        print(f"[PAD] {self.name} → {self.state.debug_summary()}")
        return resp

    def _update_state_with_structured_delta(
        self,
        *,
        situation: str,
        latest_message: str,
        partner_state_text: str | None,
        environment: str = "",
    ) -> None:
        msg_snippet = " ".join((latest_message or "").split())[:200]
        partner_line = (partner_state_text or "Unknown").strip()
        env_line = environment.strip() or "Unknown"
        prompt = (
            f"State: {self.state.debug_summary()}\n"
            f"Partner: {partner_line}\n"
            f"Env: {env_line}\n"
            f"Message: {msg_snippet or '(empty)'}\n"
            "Output JSON only, keys: pleasure arousal dominance trust interest receptivity momentum, values -0.3 to 0.3"
        )
        delta = None
        try:
            raw = self._state_delta_llm.complete(prompt, json_mode=True)
            delta = self._parse_state_delta(raw)
            if delta is None:
                fallback_raw = self._state_delta_llm.complete(prompt, json_mode=False)
                delta = self._parse_state_delta(fallback_raw)
        except Exception as e:
            self._delta_fallbacks = getattr(self, "_delta_fallbacks", 0) + 1
            if self._delta_fallbacks <= 3:
                print(f"[PAD-DELTA FAIL] {type(e).__name__}: {e}")
            return
        if delta:
            self.state.apply_delta(delta)

    def _parse_state_delta(self, raw: str | None) -> dict[str, float] | None:
        if not raw:
            return None
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return None
        out: dict[str, float] = {}
        for field in self._STATE_FIELDS:
            val = data.get(field)
            if not isinstance(val, (int, float)):
                return None
            out[field] = float(max(-0.3, min(0.3, float(val))))
        return out

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
