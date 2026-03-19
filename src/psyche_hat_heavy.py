from __future__ import annotations

"""
Heavy PsycheHat (optional):
- ChromaDB persistent vector store (RAG over past wins)
- Torch MLP that trains on wins and persists weights
- Triple-score memory retrieval + reflection (Generative Agents inspired)

This module is intentionally optional because it adds heavyweight deps.
If imports fail, instantiating `PsycheHatHeavy()` raises a clear error.
"""

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any


_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import chromadb  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e

from src.agent import LlmAgent, LlmConfig


WEIGHTS_PATH = "./psyche_memory/mlp_weights.pt"

APPROACHES = [
    "rapport first + light touch",
    "dark humor opener",
    "playful teasing",
    "genuine vulnerability",
    "mysterious confident",
    "dry sarcasm",
    "weird observation",
    "deep shared interest",
    "callback to earlier moment",
    "light physical presence",
]


@dataclass
class MemoryObject:
    desc: str
    created: int
    last_accessed: int
    importance: int
    embedding: "np.ndarray"


class PsycheHatHeavy:
    def __init__(self):
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(
                "Heavy PsycheHat dependencies are missing.\n"
                "Install: pip install torch chromadb numpy\n"
                "And pull embeddings model: ollama pull nomic-embed-text\n"
                f"Original import error: {_IMPORT_ERROR}"
            ) from _IMPORT_ERROR

        os.makedirs("./psyche_memory", exist_ok=True)
        self.client = chromadb.PersistentClient(path="./psyche_memory")
        self.collection = self.client.get_or_create_collection("social_tactics")

        # MLP: embed_dim(768) → hidden → predicted success (0..1)
        self.mlp = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        if os.path.exists(WEIGHTS_PATH):
            self.mlp.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

        self.memory_size: int = self.collection.count()
        self.ab_stats: dict[str, list[int]] = {"with_hat": [], "without_hat": []}
        self.memory_stream: list[MemoryObject] = []
        self._sim_tick = 0
        self._importance_since_reflect = 0
        self._reflect_llm = LlmAgent(
            llm=LlmConfig(temperature=0.35, num_predict=220)
        )
        self._mem_since_reflect = 0

    def embed(self, text: str) -> "np.ndarray":
        from ollama import embeddings

        resp = embeddings(model="nomic-embed-text", prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)

    def _cosine(self, a: "np.ndarray", b: "np.ndarray") -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def _add_memory(self, desc: str, importance: int, emb: "np.ndarray") -> None:
        importance = max(1, int(importance))
        vec = emb.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        mem = MemoryObject(
            desc=desc.strip(),
            created=self._sim_tick,
            last_accessed=self._sim_tick,
            importance=importance,
            embedding=vec,
        )
        self.memory_stream.append(mem)
        self._sim_tick += 1
        self._importance_since_reflect += importance
        self._mem_since_reflect += 1
        self._maybe_reflect()

    def _retrieve_memories(self, query: str, k: int = 8) -> list[MemoryObject]:
        if not self.memory_stream:
            return []
        q_emb = self.embed(query)
        norm = np.linalg.norm(q_emb)
        if norm:
            q_emb = q_emb / norm
        scored: list[tuple[float, MemoryObject]] = []
        for mem in self.memory_stream:
            recency_delta = max(0, self._sim_tick - mem.last_accessed)
            recency = 0.995 ** recency_delta
            relevance = max(0.0, self._cosine(q_emb, mem.embedding))
            score = recency * max(1, mem.importance) * relevance
            scored.append((score, mem))
        top_pairs = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        top = [mem for score, mem in top_pairs if score > 0]
        for mem in top:
            mem.last_accessed = self._sim_tick
        return top

    def _maybe_reflect(self) -> None:
        if not self.memory_stream:
            return
        if self._importance_since_reflect < 150 and self._mem_since_reflect < 32:
            return
        self._importance_since_reflect = 0
        self._mem_since_reflect = 0
        self._run_reflection()

    def _run_reflection(self) -> None:
        top = sorted(self.memory_stream, key=lambda m: m.importance, reverse=True)[:100]
        if not top:
            return
        context = "\n".join(f"- {m.desc}" for m in top)
        prompt = (
            "You synthesize social interaction memories. Given the entries below, "
            "list 3 salient high-level questions we can answer. Return JSON array of strings.\n"
            f"Memories:\n{context}\n"
        )
        raw = self._reflect_llm.complete(prompt, json_mode=True)
        try:
            questions = json.loads(raw)
        except Exception:
            questions = []
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()]
        for q in questions[:3]:
            relevant = self._retrieve_memories(q, k=5)
            if not relevant:
                continue
            refs = "\n".join(f"- {m.desc}" for m in relevant)
            insight_prompt = (
                "Given the following question and supporting memories, craft one high-level insight "
                "(2 sentences max) citing the specific patterns observed.\n"
                f"Question: {q}\nMemories:\n{refs}\nInsight:"
            )
            insight = self._reflect_llm.complete(insight_prompt).strip()
            if not insight:
                continue
            emb = self.embed(insight)
            self._add_memory(f"Reflection: {insight}", importance=9, emb=emb)

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
        used_hat: bool = False,
    ):
        if int(score) < int(min_score):
            return

        key = "with_hat" if used_hat else "without_hat"
        self.ab_stats[key].append(int(score))

        embed_text = (
            f"Scenario: {scenario} | Goal: {goal} | "
            f"You: {you_traits} | Target: {target_traits} | Approach: {approach}"
        )
        emb = self.embed(embed_text)

        self.collection.add(
            documents=[embed_text],
            embeddings=[emb.tolist()],
            metadatas=[
                {
                    "scenario": scenario[:200],
                    "goal": goal[:100],
                    "you_traits": you_traits[:100],
                    "target_traits": target_traits[:100],
                    "approach": approach,
                    "score": int(score),
                    "used_hat": str(bool(used_hat)),
                    "history_preview": str(history[:2]),
                    "timestamp": time.time(),
                }
            ],
            ids=[f"win_{self.memory_size}"],
        )
        self.memory_size += 1

        x = torch.tensor(emb).unsqueeze(0)
        y = torch.tensor([[int(score) / 100.0]], dtype=torch.float32)
        loss = self.criterion(self.mlp(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        torch.save(self.mlp.state_dict(), WEIGHTS_PATH)

        memory_desc = (
            f"Winner ({approach}) score {score}/100 — Scenario='{scenario[:80]}' Target='{target_traits[:60]}'"
        )
        importance = max(1, round(int(score) / 10))
        self._add_memory(memory_desc, importance=importance, emb=emb)

    def get_guidance(
        self,
        *,
        scenario: str,
        goal: str,
        you_traits: str,
        target_traits: str,
    ) -> dict[str, Any]:
        query = (
            f"Scenario: {scenario} | Goal: {goal} | "
            f"You: {you_traits} | Target: {target_traits}"
        )
        q_emb = self.embed(query)

        n = min(3, max(1, self.memory_size))
        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=n)
        metas = (results.get("metadatas") or [[]])[0]
        mem_hits = self._retrieve_memories(query, k=3)

        x = torch.tensor(q_emb).unsqueeze(0)
        with torch.no_grad():
            predicted = float(self.mlp(x).item() * 100)

        reflection_tip = mem_hits[0].desc if mem_hits else ""

        if metas:
            best = max(metas, key=lambda m: m.get("score", 0))
            approach = best.get("approach", random.choice(APPROACHES))
            hist_score = best.get("score", 0)
            tip = (
                f"PsycheHatHeavy [{self.memory_size} wins]: start with '{approach}' — "
                f"{hist_score}% on similar cases. MLP estimate: {round(predicted)}%."
            )
        else:
            approach = random.choice(APPROACHES)
            tip = f"PsycheHatHeavy [cold start]: try '{approach}' as opener."

        if reflection_tip:
            tip += f" Memory: {reflection_tip}"

        return {
            "recommended_approach": approach,
            "predicted_success": round(predicted, 1),
            "past_wins": metas,
            "tip": tip,
            "memory_hits": [m.desc for m in mem_hits],
        }

