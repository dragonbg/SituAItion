from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Win:
    scenario: str
    goal: str
    you_traits: str
    target_traits: str
    approach: str
    score: int
    history: list[str]
    created_at: str


class PsycheHat:
    """
    Lightweight persistent memory of past "wins".

    This is intentionally simple: store structured winners on disk and retrieve
    the closest-looking ones with a cheap similarity heuristic.
    (We can upgrade to embeddings/Chroma later once the simulator itself is stable.)
    """

    def __init__(self, path: str = "./psyche_memory/wins.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._wins: list[Win] = []
        self._load()

    @property
    def memory_size(self) -> int:
        return len(self._wins)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._wins.append(
                        Win(
                            scenario=str(obj.get("scenario", "")),
                            goal=str(obj.get("goal", "")),
                            you_traits=str(obj.get("you_traits", "")),
                            target_traits=str(obj.get("target_traits", "")),
                            approach=str(obj.get("approach", "")),
                            score=int(obj.get("score", 0)),
                            history=list(obj.get("history", [])),
                            created_at=str(obj.get("created_at", "")),
                        )
                    )
                except Exception:
                    continue

    def store_success(
        self,
        *,
        scenario: str,
        goal: str,
        you_traits: str,
        target_traits: str,
        approach: str,
        score: int,
        history: list[str],
        min_score: int = 70,
    ) -> None:
        if int(score) < int(min_score):
            return
        win = Win(
            scenario=scenario,
            goal=goal,
            you_traits=you_traits,
            target_traits=target_traits,
            approach=approach,
            score=int(score),
            history=history,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        self._wins.append(win)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "scenario": win.scenario,
                        "goal": win.goal,
                        "you_traits": win.you_traits,
                        "target_traits": win.target_traits,
                        "approach": win.approach,
                        "score": win.score,
                        "history": win.history[:24],
                        "created_at": win.created_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def get_guidance(self, *, scenario: str, goal: str, you_traits: str, target_traits: str) -> dict[str, Any]:
        """
        Return a simple tip + a recommended approach from similar past wins.
        Similarity is a cheap token overlap heuristic (fast, no extra deps).
        """
        if not self._wins:
            return {"recommended_approach": "", "tip": "", "past_wins": []}

        q = f"{scenario} {goal} {you_traits} {target_traits}".lower()
        q_tokens = {t for t in q.replace("|", " ").replace("\n", " ").split() if len(t) > 3}

        def sim(w: Win) -> int:
            text = f"{w.scenario} {w.goal} {w.you_traits} {w.target_traits}".lower()
            toks = {t for t in text.replace("|", " ").replace("\n", " ").split() if len(t) > 3}
            return len(q_tokens & toks)

        ranked = sorted(self._wins, key=lambda w: (sim(w), w.score), reverse=True)
        top = ranked[:3]
        best = top[0]
        tip = f"PsycheHat ({self.memory_size} wins): '{best.approach}' worked well on a similar case (score {best.score}/100)."
        return {
            "recommended_approach": best.approach,
            "tip": tip,
            "past_wins": [
                {"approach": w.approach, "score": w.score, "created_at": w.created_at} for w in top
            ],
        }

