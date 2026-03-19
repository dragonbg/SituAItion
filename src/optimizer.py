from __future__ import annotations

import json
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any

try:  # Optional – only present when Groq backend is installed
    from groq import BadRequestError as _GroqBadRequestError  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    class _GroqBadRequestError(Exception):
        pass

from tqdm import tqdm

from src.agent import LlmAgent, LlmConfig
from src.psyche_hat import PsycheHat

LOG_PROGRESS = os.getenv("SITUAITION_LOG_PROGRESS", "").strip().lower() in {"1", "true", "yes"}
_BACKEND = os.getenv("SITUAITION_BACKEND", "").strip().lower()
_MAX_CONCURRENT_REQUESTS = 20 if _BACKEND == "groq" else 0
_REQUEST_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_REQUESTS) if _MAX_CONCURRENT_REQUESTS else None


@contextmanager
def _request_slot():
    if _REQUEST_SEMAPHORE is None:
        yield
        return
    _REQUEST_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _REQUEST_SEMAPHORE.release()


def _log_progress(msg: str) -> None:
    if LOG_PROGRESS:
        print(f"[optimizer] {msg}", flush=True)


@dataclass(frozen=True)
class CandidatePlan:
    title: str
    steps: list[str]
    exact_words: str
    fallback_if_no: str
    notes: str = ""


@dataclass(frozen=True)
class ScoredPlan:
    plan: CandidatePlan
    score: int
    judge_notes: str


_MANIPULATION_RED_FLAGS = [
    # Keep this list narrow to avoid penalizing legitimate wording.
    "force",
    "trick",
    "guilt",
    "lie",
    "fake",
    "blackmail",
]


def _red_flag_penalty(text: str) -> int:
    t = (text or "").lower()
    hits = sum(1 for w in _MANIPULATION_RED_FLAGS if w in t)
    return hits * 12


def _extract_int(text: str) -> int:
    m = re.search(r"\b(\d{1,3})\b", text or "")
    if not m:
        return 0
    return max(0, min(100, int(m.group(1))))


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def generate_candidate_plans(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    n: int = 12,
    llm: LlmAgent | None = None,
) -> list[CandidatePlan]:
    agent = llm or LlmAgent()

    prompt = f"""
You are generating social interaction plans.
Hard constraints:
- Be honest, consent-first, and low-pressure.
- No deception, no "micro-timed" manipulation scripts, no covert tactics.
- Offer an easy out and respect "no" immediately.
- Keep steps practical and human.

Return ONLY valid JSON: an array of {n} objects with keys:
title (string), steps (array of 3-7 strings), exact_words (string),
fallback_if_no (string), notes (string).

Scenario: {scenario}
Goal: {goal}
You traits: {you_traits}
Target traits: {target_traits}
""".strip()

    raw = agent.complete(prompt, json_mode=True)
    data = _safe_json_loads(raw)
    if not isinstance(data, list):
        # Minimal fallback: wrap raw text as one candidate.
        return [
            CandidatePlan(
                title="Simple respectful ask",
                steps=[
                    "Start with a brief, friendly opener relevant to the moment.",
                    "State your ask clearly with no pressure.",
                    "Offer an easy out and accept the answer.",
                ],
                exact_words=raw[:400] if raw else "Hey—quick question. Would you be comfortable sharing that? No worries if not.",
                fallback_if_no="All good—thanks anyway. (Then change topic / exit gracefully.)",
                notes="LLM did not return JSON; used fallback.",
            )
        ]

    plans: list[CandidatePlan] = []
    for item in data[:n]:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps") if isinstance(item.get("steps"), list) else []
        steps = [str(s).strip() for s in steps if str(s).strip()]
        plans.append(
            CandidatePlan(
                title=str(item.get("title", "")).strip() or "Untitled plan",
                steps=steps[:7] if steps else ["Ask clearly and respectfully, with an easy out."],
                exact_words=str(item.get("exact_words", "")).strip()
                or "Hey—quick question. Would you be comfortable with that? No worries if not.",
                fallback_if_no=str(item.get("fallback_if_no", "")).strip()
                or "No problem at all—thanks for being straight with me.",
                notes=str(item.get("notes", "")).strip(),
            )
        )

    return plans or [
        CandidatePlan(
            title="Simple respectful ask",
            steps=["Ask clearly and respectfully, with an easy out."],
            exact_words="Hey—quick question. Would you be comfortable with that? No worries if not.",
            fallback_if_no="No problem at all—thanks anyway.",
            notes="No usable candidates parsed; used fallback.",
        )
    ]


def judge_plan(
    *,
    scenario: str,
    goal: str,
    plan: CandidatePlan,
    llm: LlmAgent | None = None,
) -> ScoredPlan:
    agent = llm or LlmAgent(llm=LlmConfig(temperature=0.2, num_predict=120))

    prompt = f"""
You are judging a proposed social interaction plan.
Score 0-100 for: (a) likelihood of a positive outcome, (b) respect/consent, (c) honesty, (d) low pressure.
Penalize deception, pressure, coercion, or "tactics" that bypass consent.
Output ONLY a JSON object with keys: score (int 0-100), notes (string).

Scenario: {scenario}
Goal: {goal}
Plan title: {plan.title}
Steps: {plan.steps}
Exact words: {plan.exact_words}
Fallback if no: {plan.fallback_if_no}
""".strip()

    try:
        raw = agent.complete(prompt, json_mode=True)
    except _GroqBadRequestError as err:  # type: ignore[arg-type]
        if "json_validate_failed" not in str(err):
            raise
        raw = agent.complete(prompt, json_mode=False)
        parsed = _try(raw)
    else:
        parsed = _try(raw)
    if parsed is None:
        raw2 = agent.complete(prompt, json_mode=False)
        parsed = _try(raw2)

    if parsed is None:
        fallback_text = raw or raw2 or ""
        extracted = _extract_int(fallback_text)
        score = extracted if extracted != 0 else 50
        notes = ""
    else:
        score, notes = parsed

    text_blob = " ".join([plan.title, " ".join(plan.steps), plan.exact_words, plan.fallback_if_no, plan.notes])
    score = max(0, min(100, int(score) - _red_flag_penalty(text_blob)))
    return ScoredPlan(plan=plan, score=score, judge_notes=notes or (raw[:400] if raw else ""))


@dataclass(frozen=True)
class SimState:
    history: list[str]  # alternating You/Target messages
    score: int = 0
    notes: str = ""
    approach: str = ""


def _json_array_of_strings_or_none(raw: str) -> list[str] | None:
    data = _safe_json_loads(raw)
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for x in data:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out or None


def propose_next_messages(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    history: list[str],
    k: int,
    llm: LlmAgent,
) -> list[str]:
    # Keep prompt bounded
    recent = "\n".join(history[-8:]) if history else "(start)"
    turn_hint = 1 + (sum(1 for h in history if h.startswith("You:")) if history else 0)
    early_turn = turn_hint <= 1
    prompt = f"""
Generate {k} different next messages for "You" in a social interaction.
Constraints:
- Natural, human, short (1-2 sentences).
- Honest, respectful, low-pressure.
- No manipulation, no deception, no "timed tactics", no persistence after a clear no.
- You can warm up with light context before asking, if appropriate.
- Avoid repeating the same wording across candidates.
- Vary intent across candidates: some should be small-talk / rapport, some a segue, some a direct low-pressure ask (depending on turn).
- IMPORTANT: If this is an early turn, do NOT ask for IG/contact.

Return ONLY valid JSON: an array of {k} strings. No other text.

Scenario: {scenario}
Goal: {goal}
You traits: {you_traits}
Target traits: {target_traits}
This is turn {turn_hint} for You.
If turn=1: ONLY rapport/opening (no contact info ask).
If turn>=2: you may segue into the ask.
Conversation so far (most recent last):
{recent}
""".strip()
    def _parse_msgs(raw_text: str) -> list[str] | None:
        msgs0 = _json_array_of_strings_or_none(raw_text)
        if msgs0 is None:
            return None
        # normalize + dedupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for m in msgs0:
            m2 = " ".join(m.replace("\uFFFD", "").split()).strip()
            if not m2:
                continue
            key = m2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(m2)
        return out or None

    raw = llm.complete(prompt, json_mode=True)
    msgs = _parse_msgs(raw)

    # Retry once without JSON mode (some models behave worse with format=json)
    if msgs is None:
        raw2 = llm.complete(prompt, json_mode=False)
        msgs = _parse_msgs(raw2)

    if msgs is None:
        # crude recovery: split lines
        msgs = [line.strip("- ").strip() for line in (raw or "").splitlines() if line.strip()]

    msgs = (msgs or [])[:k]

    # Ensure we always have k distinct options so branching actually branches.
    if len(msgs) < k:
        # Fillers are intentionally split by turn so we don't collapse into "ask immediately" branches.
        rapport_fillers = [
            "Hey—how’s your day going so far?",
            "That’s a good point—how did you get into that?",
            "Wait, I have to ask—what’s your go-to thing to do after work?",
            "Random, but I like your vibe—how’s your week been?",
            "I’m grabbing a quick break—what are you up to today?",
        ]
        ask_fillers = [
            "By the way—are you on Instagram? If you’re comfortable, I’d love to follow you.",
            "Quick one—do you use IG? Totally fine if you’d rather not share.",
            "If you’re comfortable, want to swap Instagrams?",
            "No pressure at all—what’s your Instagram?",
            "If that’s not your thing, we can swap numbers instead—up to you.",
        ]
        fillers = rapport_fillers if early_turn else ask_fillers
        random.shuffle(fillers)
        for f in fillers:
            if len(msgs) >= k:
                break
            if f.lower() not in {m.lower() for m in msgs}:
                msgs.append(f)

    # Hard guard: on turn 1, strip any candidate that asks for IG/contact.
    if early_turn:
        blocked = ("instagram", " ig", "insta", "number", "phone", "snap", "@")
        msgs = [m for m in msgs if not any(b in m.lower() for b in blocked)]
        if len(msgs) < k:
            # Refill with rapport-only fillers to reach k
            rapport_fillers = [
                "Hey—how’s your day going so far?",
                "That’s a good point—how did you get into that?",
                "Wait, I have to ask—what’s your go-to thing to do after work?",
                "Random, but I like your vibe—how’s your week been?",
                "I’m grabbing a quick break—what are you up to today?",
            ]
            random.shuffle(rapport_fillers)
            for f in rapport_fillers:
                if len(msgs) >= k:
                    break
                if f.lower() not in {m.lower() for m in msgs}:
                    msgs.append(f)

    return msgs[:k]


def simulate_target_reply(
    *,
    scenario: str,
    you_traits: str,
    target_traits: str,
    history: list[str],
    llm: LlmAgent,
) -> str:
    recent = "\n".join(history[-8:]) if history else "(start)"
    last_you = ""
    for line in reversed(history):
        if line.startswith("You:"):
            last_you = line
            break
    prompt = f"""
Roleplay the "Target" replying naturally to the last message.
Constraints:
- Target is a normal person (not an assistant).
- Keep it short (1-2 sentences).
- If the ask is uncomfortable, Target can say no or deflect.

Return ONLY the target's next message (no quotes, no extra text).

Scenario: {scenario}
You traits: {you_traits}
Target traits: {target_traits}
Last message from You: {last_you}
Conversation so far:
{recent}
""".strip()
    reply = (llm.complete(prompt) or "").replace("\uFFFD", "").strip()
    if reply:
        return reply

    fallbacks = [
        "Sorry—I'd rather not share.",
        "Uh, I don’t really give that out—sorry.",
        "Maybe another time, but thanks for asking.",
    ]
    return random.choice(fallbacks)


def judge_conversation(
    *,
    scenario: str,
    goal: str,
    history: list[str],
    llm: LlmAgent,
) -> tuple[int, str]:
    recent = "\n".join(history[-12:]) if history else "(start)"
    prompt = f"""
Judge this conversation for:
- Likelihood of a positive outcome toward the goal
- Respect/consent/honesty/low-pressure (must be high)
- Bonus points if the approach is unexpected, non-generic, or uses environmental/context clues creatively (not just polite script)
- Deduct heavily if the same ask (especially IG/contact) is repeated; a duplicate Instagram ask should cost at least 25 points
- Deduct at least 20 points if "You" confuses identities, refers to themself as the Target, or reuses the Target's handle

If there is pressure, deception, or ignoring a "no", score should drop sharply.
Output ONLY JSON with keys: score (int 0-100), notes (string).

Scenario: {scenario}
Goal: {goal}
Conversation (recent):
{recent}
""".strip()

    def _try(raw_text: str) -> tuple[int, str] | None:
        data0 = _safe_json_loads(raw_text)
        if not isinstance(data0, dict):
            return None
        sc = data0.get("score")
        if not isinstance(sc, int):
            return None
        sc = max(0, min(100, sc))
        nt = data0.get("notes")
        nt_s = str(nt).strip() if nt is not None else ""
        return sc, nt_s

    try:
        raw = llm.complete(prompt, json_mode=True)
    except _GroqBadRequestError as err:  # type: ignore[arg-type]
        if "json_validate_failed" not in str(err):
            raise
        raw = llm.complete(prompt, json_mode=False)
        parsed = _try(raw)
    else:
        parsed = _try(raw)
    if parsed is None:
        raw2 = llm.complete(prompt, json_mode=False)
        parsed = _try(raw2)

    if parsed is None:
        fallback_text = raw or raw2 or ""
        extracted = _extract_int(fallback_text)
        score = extracted if extracted != 0 else 50
        notes = ""
    else:
        score, notes = parsed

    text_blob = " ".join(history[-12:])
    score = int(score)
    score -= _red_flag_penalty(text_blob)
    score -= _repetition_penalty(history)
    score -= _identity_confusion_penalty(history)
    score += _novelty_bonus(history)
    score = max(0, min(100, score))
    return score, notes or (raw[:250] if raw else "")


def _heuristic_reward(goal: str, history: list[str]) -> int:
    """
    Cheap reward shaping so branches don't all tie.
    This is NOT ground truth—just a nudge to separate obviously-better paths.
    """
    g = (goal or "").lower()
    convo = " ".join(history[-8:]).lower()

    reward = 0
    # Positive signals
    if "instagram" in g or "ig" in g:
        if "@" in convo or "add me" in convo or "here it is" in convo or "i'll dm" in convo:
            reward += 35
        if "sure" in convo or "yeah" in convo or "of course" in convo:
            reward += 10
        if "maybe" in convo or "another time" in convo:
            reward += 4

    # Negative/stop signals
    if "i’d rather not" in convo or "i'd rather not" in convo or "no" in convo or "not comfortable" in convo:
        reward -= 15
    if "i'm not on instagram" in convo or "im not on instagram" in convo:
        reward -= 8

    # Penalize repetition (your exact complaint)
    last_you = [h for h in history if h.startswith("You:")][-3:]
    if len(set(last_you)) < len(last_you):
        reward -= 10

    return max(-25, min(35, reward))


def _novelty_bonus(history: list[str]) -> int:
    """Lightweight novelty heuristic to reward non-generic approaches."""
    text = " ".join(history[-10:]).lower()
    bonus = 0

    env_words = (
        "rooftop",
        "cafe",
        "bar",
        "window",
        "playlist",
        "lighting",
        "weather",
        "crowd",
        "bartender",
        "art",
        "book",
        "coffee",
        "earbuds",
        "balcony",
    )
    if any(w in text for w in env_words):
        bonus += 3

    playful_signals = (
        "callback",
        "inside joke",
        "plot twist",
        "curveball",
        "random dare",
        "coin flip",
        "bet you",
        "tease",
        "disqualify",
    )
    if any(sig in text for sig in playful_signals):
        bonus += 3

    if "(pause" in text or "[pause" in text or "stage" in text:
        bonus += 2

    unique_messages = {line.strip().lower() for line in history if line.startswith("You:")}
    if len(unique_messages) >= 3:
        bonus += 1

    return min(8, bonus)


def _repetition_penalty(history: list[str]) -> int:
    """Penalize repeating the same ask or keyword over and over."""
    you_lines = [line.split(":", 1)[1].strip().lower() for line in history if line.startswith("You:")]
    if not you_lines:
        return 0

    normalized_counts: dict[str, int] = {}
    for msg in you_lines:
        cleaned = re.sub(r"[^a-z0-9@ ]+", " ", msg)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        normalized_counts[cleaned] = normalized_counts.get(cleaned, 0) + 1

    penalty = sum((count - 1) * 15 for count in normalized_counts.values() if count > 1)

    ig_keywords = ("instagram", "insta", "ig", "@")
    contact_keywords = ig_keywords + (
        "phone",
        "number",
        "snap",
        "snapchat",
        "tiktok",
        "handle",
        "contact",
        "dm",
        "text you",
        "whatsapp",
    )

    contact_hits = 0
    ig_hits = 0
    for msg in you_lines:
        if any(kw in msg for kw in contact_keywords):
            contact_hits += 1
        if any(kw in msg for kw in ig_keywords):
            ig_hits += 1

    if ig_hits > 1:
        penalty = max(penalty, 25 + max(0, ig_hits - 2) * 10)
    elif contact_hits > 1:
        penalty += (contact_hits - 1) * 12

    return min(70, penalty)


def _identity_confusion_penalty(history: list[str]) -> int:
    """Penalize when the "You" agent impersonates or borrows the Target identity."""
    target_names: set[str] = set()
    target_handles: set[str] = set()
    for line in history:
        if not line.startswith("Target:"):
            continue
        content = line.split(":", 1)[1].strip()
        for match in re.findall(r"(?:i'm|i am|my name is|this is)\s+([A-Za-z][A-Za-z\-']{1,})", content, flags=re.IGNORECASE):
            target_names.add(match.lower())
        for handle in re.findall(r"@[A-Za-z0-9_.-]+", content):
            target_handles.add(handle.lower())

    penalty = 0
    confused = False
    you_lines = [line.split(":", 1)[1].strip() for line in history if line.startswith("You:")]
    for msg in you_lines:
        lower = msg.lower()
        if lower.startswith("target:"):
            penalty += 15
            confused = True
        if "target here" in lower or "this is target" in lower:
            penalty += 12
            confused = True
        for name in target_names:
            if re.search(rf"(?:i'm|i am|this is|it's)\s+{re.escape(name)}\b", lower):
                penalty += 12
                confused = True
                break
        for handle in target_handles:
            if handle in lower:
                penalty += 12
                confused = True
                break

    if confused:
        penalty = max(20, penalty)

    return min(60, penalty)


def beam_search_simulation(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    turns: int = 4,
    branch_factor: int = 6,
    beam_width: int = 12,
    llm: LlmAgent | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """
    Multi-turn branching simulation (beam search):
    - expand each beam state into branch_factor candidate next "You" messages
    - simulate a "Target" reply for each
    - keep top beam_width states (cheap heuristic per expansion)
    - run LLM judging only on the beam states each turn (much faster)
    """
    def _progress(p: float, desc: str):
        if progress is None:
            return
        try:
            progress(max(0.0, min(1.0, float(p))), desc=desc)
        except TypeError:
            progress(max(0.0, min(1.0, float(p))), desc)

    # Use lightweight configs; branching search needs many short calls.
    proposer_llm = llm or LlmAgent(llm=LlmConfig(temperature=0.9, num_predict=120))
    actor_llm = LlmAgent(llm=LlmConfig(temperature=0.8, num_predict=140))
    judge_llm = LlmAgent(llm=LlmConfig(temperature=0.15, num_predict=120))

    approach_library = [
        "warm opener + situational comment",
        "shared interest probe",
        "light humor + curiosity",
        "direct but low-pressure ask",
        "slow build: small talk then segue",
    ]
    beams: list[SimState] = [SimState(history=[], approach=random.choice(approach_library))]
    total_expansions = max(1, turns) * max(1, beam_width) * max(1, branch_factor)
    done = 0

    for t in range(turns):
        _progress(0.01 + 0.02 * (t / max(1, turns)), f"Expanding turn {t+1}/{turns}...")
        candidates: list[SimState] = []
        # Cross-beam dedup: if different beam states propose the same "You" message,
        # keep only the first to avoid wasting compute on identical expansions.
        seen_you_msgs: set[str] = set()

        for b_idx, state in enumerate(beams, start=1):
            you_msgs = propose_next_messages(
                scenario=scenario,
                goal=goal,
                you_traits=you_traits,
                target_traits=target_traits,
                history=state.history,
                k=branch_factor,
                llm=proposer_llm,
            )
            for m_idx, you_msg in enumerate(you_msgs, start=1):
                key = " ".join((you_msg or "").lower().split())
                if key in seen_you_msgs:
                    continue
                seen_you_msgs.add(key)

                done += 1
                _progress(
                    0.05 + 0.90 * (done / total_expansions),
                    f"Simulating (turn {t+1}/{turns}) branch {m_idx}/{len(you_msgs)} of beam {b_idx}/{len(beams)}...",
                )
                new_history = [*state.history, f"You: {you_msg}"]
                target_msg = simulate_target_reply(
                    scenario=scenario,
                    you_traits=you_traits,
                    target_traits=target_traits,
                    history=new_history,
                    llm=actor_llm,
                )
                new_history.append(f"Target: {target_msg}")

                # Cheap heuristic during expansion: prefer branches that don't trigger red flags
                # and look like an actual conversation (not empty/garbled).
                text_blob = " ".join(new_history[-4:])
                heuristic = 60
                heuristic -= _red_flag_penalty(text_blob)
                if len(you_msg) < 4 or len(target_msg) < 2:
                    heuristic -= 10
                heuristic += _heuristic_reward(goal, new_history)
                candidates.append(
                    SimState(
                        history=new_history,
                        score=max(0, min(100, heuristic)),
                        notes="",
                        approach=state.approach,
                    )
                )

        # Keep best few branches
        candidates.sort(key=lambda s: s.score, reverse=True)
        beams = candidates[: max(1, beam_width)]

        # Now do the expensive LLM judging only on the surviving beam states.
        judged_beams: list[SimState] = []
        for i, s in enumerate(beams, start=1):
            _progress(
                0.10 + 0.85 * ((t + i / max(1, len(beams))) / max(1, turns)),
                f"Judging top branches (turn {t+1}/{turns}) {i}/{len(beams)}...",
            )
            score, notes = judge_conversation(
                scenario=scenario,
                goal=goal,
                history=s.history,
                llm=judge_llm,
            )
            blended = round(0.65 * score + 0.35 * s.score)
            judged_beams.append(SimState(history=s.history, score=blended, notes=notes, approach=s.approach))

        judged_beams.sort(key=lambda s: s.score, reverse=True)
        beams = judged_beams[: max(1, beam_width)]

    best = max(beams, key=lambda s: s.score)
    _progress(0.99, "Finalizing best branch...")
    return {
        "best": {
            "score": best.score,
            "history": best.history,
            "judge_notes": best.notes,
        },
        "alternatives": [
            {"score": s.score, "history": s.history, "judge_notes": s.notes} for s in beams[1:3]
        ],
        "meta": {
            "turns": turns,
            "branch_factor": branch_factor,
            "beam_width": beam_width,
        },
    }


def _rollout_one(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    turns: int,
    approach: str,
    seed_history: list[str] | None,
    proposer_llm: LlmAgent,
    actor_llm: LlmAgent,
    progress_cb: Any | None = None,
    progress_base: float = 0.0,
    progress_span: float = 0.0,
    use_generative_agents: bool = False,
) -> list[str]:
    history: list[str] = list(seed_history or [])

    def _p_local(step_frac: float, desc: str):
        if progress_cb is None or progress_span <= 0:
            return
        try:
            progress_cb(progress_base + progress_span * step_frac, desc=desc)
        except Exception:
            return

    def write_next_message() -> str:
        # Faster than propose_next_messages(k=1): no JSON, shorter output.
        recent = "\n".join(history[-8:]) if history else "(start)"
        you_turn = 1 + sum(1 for h in history if h.startswith("You:"))
        early = you_turn <= 1
        prompt = f"""
Write the next message for "You" (1–2 sentences).
Constraints:
- Natural, specific, human.
- No deception or coercion; accept a clear no.
- If this is turn 1: rapport only (no asking for IG/number).
- If turn>=2: you may segue toward the goal, low-pressure.

Scenario: {scenario}
Goal: {goal}
You traits: {you_traits}
Target traits: {target_traits}
Approach style hint: {approach}
Conversation so far:
{recent}
""".strip()
        msg = (proposer_llm.complete(prompt, json_mode=False) or "").replace("\uFFFD", "").strip()
        if not msg:
            return "Hey—how’s your day going so far?" if early else "By the way—are you on Instagram? No worries if not."
        # Strip accidental bullets/quotes
        msg = msg.strip().strip('"').strip()
        return msg

    # Optional: use Stanford-style agents (memory/reflection) during rollouts.
    you_agent = None
    target_agent = None
    if use_generative_agents:
        from src.agent import GenerativeAgent

        you_agent = GenerativeAgent(
            name="You",
            traits=you_traits,
            goal=goal,
            scenario=scenario,
            target_traits=target_traits,
            psyche_hat=None,
            llm=proposer_llm,
        )
        target_agent = GenerativeAgent(
            name="Target",
            traits=target_traits,
            goal="Respond naturally; be open if comfortable, deflect if not.",
            scenario=scenario,
            target_traits=target_traits,
            psyche_hat=None,
            llm=actor_llm,
        )

    for t in range(max(1, int(turns))):
        _p_local((t + 0.05) / max(1, int(turns)), f"Sim turn {t+1}/{turns} — drafting your message...")
        if you_agent is not None:
            situation = f"Turn {t+1}. Scenario: {scenario}"
            partner_text = target_agent.state.to_text() if target_agent is not None else None
            partner_receptivity = target_agent.state.receptivity if target_agent is not None else None
            you_msg = you_agent.react_message(
                situation=situation,
                approach=approach,
                partner_state_text=partner_text,
                partner_receptivity=partner_receptivity,
            )
        else:
            you_msg = write_next_message()
        history.append(f"You: {you_msg}")

        _p_local((t + 0.35) / max(1, int(turns)), f"Sim turn {t+1}/{turns} — simulating target reply...")
        if target_agent is not None:
            target_msg = target_agent.react_message(
                situation=f"You received: {you_msg}", approach=""
            )
            # Update memory streams a bit
            you_agent.observe(f"Target: {target_msg}", importance=7)
            target_agent.observe(f"You: {you_msg}", importance=6)
            if (t + 1) % 3 == 0:
                you_agent.reflect()
                target_agent.reflect()
        else:
            target_msg = simulate_target_reply(
                scenario=scenario,
                you_traits=you_traits,
                target_traits=target_traits,
                history=history,
                llm=actor_llm,
            )
        history.append(f"Target: {target_msg}")
    return history


def render_micro_tactics(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    message_history: list[str],
    llm: LlmAgent | None = None,
) -> str:
    """
    Winner-only render pass: convert the winning message-level transcript into
    hyper-granular micro-tactics (timing/body language/exact words), without
    contaminating the search with long outputs.
    """
    render_model = os.getenv("SITUAITION_RENDER_MODEL") or os.getenv("SITUAITION_MODEL", "qwen3:8b")
    agent = llm or LlmAgent(llm=LlmConfig(model=render_model, temperature=0.8, num_predict=1000))

    convo = "\n".join(message_history[-16:]) if message_history else "(start)"
    prompt = f"""
You are converting a successful conversation outline into a hyper-granular "playbook".
Constraints:
- Keep it realistic for a normal human interaction.
- Consent-first and low-pressure. If the target signals discomfort or says no, the playbook must de-escalate.
- No deception or coercion.

Input transcript (message-level):
{convo}

Output:
- 6–14 numbered steps with approximate timing (t=..s), posture/eye contact/voice notes, and what to say.
- Include at least 2 "if they respond X, do Y" branches.

Scenario: {scenario}
Goal: {goal}
You traits: {you_traits}
Target traits: {target_traits}
""".strip()
    return agent.complete(prompt)


def evolutionary_search_and_render(
    *,
    scenario: str,
    goal: str,
    you_traits: str,
    target_traits: str,
    num_sims: int = 64,
    turns: int = 6,
    judge_top_k: int = 8,
    progress: Any | None = None,
    hat: PsycheHat | None = None,
    ab_test_hat: bool = False,
    use_generative_agents: bool = False,
) -> dict[str, Any]:
    """
    Phase 1/2 evolutionary search over SHORT messages.

    Phase 1: explore random approaches.
    Phase 2: branch from top winners by seeding their opening and mutating the approach.
    Then render micro-tactics ONCE for the best branch.
    """
    def _p(frac: float, desc: str):
        if progress is None:
            return
        try:
            progress(max(0.0, min(0.99, float(frac))), desc=desc)
        except Exception:
            return

    approaches = [
        "strategic pause before responding",
        "callback to environmental detail",
        "false disinterest pivot",
        "misinterpret-then-correct",
        "redirect to observation about surroundings",
        "playful challenge with quick soften",
        "sincere vulnerability burst",
    ]

    # If enabled, use PsycheHat to bias one of the approaches (A/B test).
    hat_tip = ""
    hat_rec = ""
    if hat:
        g = hat.get_guidance(scenario=scenario, goal=goal, you_traits=you_traits, target_traits=target_traits)
        hat_tip = str(g.get("tip", "")).strip()
        hat_rec = str(g.get("recommended_approach", "")).strip()
        if hat_rec and hat_rec not in approaches:
            approaches.insert(0, hat_rec)

    proposer_cfg = LlmConfig(temperature=0.95, num_predict=120)
    actor_cfg = LlmConfig(temperature=0.85, num_predict=140)
    judge_llm = LlmAgent(llm=LlmConfig(temperature=0.15, num_predict=120))

    def _run_rollout_task(approach: str, seed_history: list[str] | None) -> dict[str, Any]:
        with _request_slot():
            local_proposer = LlmAgent(llm=proposer_cfg)
            local_actor = LlmAgent(llm=actor_cfg)
            history = _rollout_one(
                scenario=scenario,
                goal=goal,
                you_traits=you_traits,
                target_traits=target_traits,
                turns=turns,
                approach=approach,
                seed_history=seed_history,
                proposer_llm=local_proposer,
                actor_llm=local_actor,
                progress_cb=None,
                progress_base=0.0,
                progress_span=0.0,
                use_generative_agents=use_generative_agents,
            )
        pre = 50 + _heuristic_reward(goal, history)
        return {"history": history, "score": int(pre), "approach": approach, "judge_notes": ""}

    results: list[dict[str, Any]] = []
    sims = max(8, int(num_sims))
    phase1 = sims // 2

    max_workers_env = os.getenv("SITUAITION_MAX_WORKERS", "4")
    try:
        max_workers_cfg = int(max_workers_env)
    except (TypeError, ValueError):  # pragma: no cover - env parsing
        max_workers_cfg = 4
    max_workers = max(1, min(max_workers_cfg, sims))

    # ── Phase 1: explore ─────────────────────────────────────────
    phase1_futures: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(phase1):
            approach = random.choice(approaches)
            if ab_test_hat and hat_rec and i % 2 == 0:
                approach = hat_rec
            phase1_futures.append(executor.submit(_run_rollout_task, approach, None))

        completed_phase1 = 0
        for future in as_completed(phase1_futures):
            payload = future.result()
            completed_phase1 += 1
            _p((completed_phase1 + 0.10) / max(1, sims), f"Phase 1 — exploring {completed_phase1}/{phase1}...")
            results.append(payload)

    # LLM-judge only the top K from phase 1 to create real separation.
    k1 = max(3, min(int(judge_top_k), len(results)))
    _p(phase1 / sims, f"Judging top {k1} candidates from Phase 1...")
    phase1_ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, r in enumerate(phase1_ranked[:k1], start=1):
        _p((phase1 + idx / (k1 + 1)) / sims, f"Phase 1 judging {idx}/{k1}...")
        s, notes = judge_conversation(scenario=scenario, goal=goal, history=r["history"], llm=judge_llm)
        r["score"] = max(0, min(100, int(s + _heuristic_reward(goal, r["history"]))))
        r["judge_notes"] = notes

    # ── Phase 2: branch from winners ─────────────────────────────
    top_k = max(3, len(results) // 5)
    winners = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    remaining = sims - phase1
    per_winner = max(1, remaining // max(1, len(winners)))

    remaining = sims - phase1
    per_winner = max(1, remaining // max(1, len(winners)))
    phase2_futures: list[Any] = []
    if winners and per_winner > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for w in winners:
                for _ in range(per_winner):
                    mutated = f"{w['approach']} + {random.choice(approaches)}"
                    seed = w["history"][:4]
                    phase2_futures.append(executor.submit(_run_rollout_task, mutated, seed))

            completed_phase2 = 0
            total_phase2 = len(phase2_futures)
            for future in as_completed(phase2_futures):
                payload = future.result()
                completed_phase2 += 1
                progress_idx = phase1 + completed_phase2
                _p((progress_idx + 0.10) / max(1, sims), f"Phase 2 — branching {completed_phase2}/{max(1, total_phase2)}...")
                results.append(payload)

    # LLM-judge only top K overall before choosing the winner.
    k2 = max(5, min(int(judge_top_k) * 2, len(results)))
    _p(0.85, f"Judging top {k2} candidates overall...")
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, r in enumerate(ranked[:k2], start=1):
        _p(0.85 + 0.10 * (idx / (k2 + 1)), f"Final judging {idx}/{k2}...")
        s, notes = judge_conversation(scenario=scenario, goal=goal, history=r["history"], llm=judge_llm)
        r["score"] = max(0, min(100, int(s + _heuristic_reward(goal, r["history"]))))
        r["judge_notes"] = notes

    best = max(ranked[:k2], key=lambda x: x["score"])
    _p(0.98, "Rendering micro-tactics for best path...")
    playbook = render_micro_tactics(
        scenario=scenario,
        goal=goal,
        you_traits=you_traits,
        target_traits=target_traits,
        message_history=best["history"],
    )

    if hat:
        hat.store_success(
            scenario=scenario,
            goal=goal,
            you_traits=you_traits,
            target_traits=target_traits,
            approach=str(best.get("approach", "")),
            score=int(best.get("score", 0)),
            history=list(best.get("history", [])),
            min_score=75,
        )

    _p(0.99, "Done.")
    return {
        "best": best,
        "playbook": playbook,
        "meta": {
            "phase1": phase1,
            "total_sims": len(results),
            "turns": turns,
            "hat_tip": hat_tip,
        },
    }


def monte_carlo_optimize(
    scenario: str,
    you_traits: str,
    target_traits: str,
    goal: str,
    *,
    candidates: int = 12,
    judges: int = 24,
    progress: Any | None = None,
) -> dict[str, Any]:
    """
    "Monte Carlo" here means: sample multiple candidate plans + evaluate them across multiple judge samples.
    It's not a guarantee—just a way to reduce variance and avoid single-shot advice.
    """
    def _progress(p: float, desc: str):
        if progress is None:
            return
        try:
            progress(max(0.0, min(1.0, float(p))), desc=desc)
        except TypeError:
            # Support callers that pass a simple callable(progress_float, desc_str)
            progress(max(0.0, min(1.0, float(p))), desc)

    llm = LlmAgent()
    _progress(0.01, "Generating candidate plans...")
    plans = generate_candidate_plans(
        scenario=scenario,
        goal=goal,
        you_traits=you_traits,
        target_traits=target_traits,
        n=candidates,
        llm=llm,
    )

    # If the model didn't return JSON and we fell back to 1 plan, don't spend forever judging.
    if len(plans) <= 1:
        judges = min(judges, 6)

    scored: list[ScoredPlan] = []

    judge_passes_per_plan = max(1, judges // max(1, len(plans)))
    total_judges = len(plans) * judge_passes_per_plan
    done = 0

    iterable = plans if progress is not None else tqdm(plans, desc="Scoring plans")
    for idx, plan in enumerate(iterable, start=1):
        _progress(0.05 + 0.05 * (idx / max(1, len(plans))), f"Preparing to score plan {idx}/{len(plans)}...")

        # Multiple judging passes to smooth randomness
        scores: list[ScoredPlan] = []
        for j in range(judge_passes_per_plan):
            done += 1
            _progress(
                0.10 + 0.85 * (done / max(1, total_judges)),
                f"Judging plan {idx}/{len(plans)} (sample {j+1}/{judge_passes_per_plan})...",
            )
            scores.append(judge_plan(scenario=scenario, goal=goal, plan=plan, llm=llm))

        avg = round(sum(s.score for s in scores) / len(scores))
        judged_notes = max(scores, key=lambda s: len(s.judge_notes or "")).judge_notes
        scored.append(ScoredPlan(plan=plan, score=avg, judge_notes=judged_notes))

    scored.sort(key=lambda x: x.score, reverse=True)
    best = scored[0]
    top3 = scored[:3]

    _progress(0.99, "Finalizing best plan...")
    return {
        "best": {
            "title": best.plan.title,
            "steps": best.plan.steps,
            "exact_words": best.plan.exact_words,
            "fallback_if_no": best.plan.fallback_if_no,
            "notes": best.plan.notes,
            "judge_notes": best.judge_notes,
            "score": best.score,
        },
        "alternatives": [
            {
                "title": s.plan.title,
                "steps": s.plan.steps,
                "exact_words": s.plan.exact_words,
                "fallback_if_no": s.plan.fallback_if_no,
                "notes": s.plan.notes,
                "judge_notes": s.judge_notes,
                "score": s.score,
            }
            for s in top3[1:]
        ],
        "meta": {
            "candidates": len(plans),
            "judges": judges,
        },
    }