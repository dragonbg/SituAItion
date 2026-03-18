from __future__ import annotations

"""
Heavy PsycheHat (optional):
- ChromaDB persistent vector store (RAG over past wins)
- Torch MLP that trains on wins and persists weights

This module is intentionally optional because it adds heavyweight deps.
If imports fail, instantiating `PsycheHatHeavy()` raises a clear error.
"""

import os
import random
import time
from typing import Any


_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import chromadb  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e


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

    def embed(self, text: str) -> "np.ndarray":
        from ollama import embeddings

        resp = embeddings(model="nomic-embed-text", prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)

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

        x = torch.tensor(q_emb).unsqueeze(0)
        with torch.no_grad():
            predicted = float(self.mlp(x).item() * 100)

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

        return {
            "recommended_approach": approach,
            "predicted_success": round(predicted, 1),
            "past_wins": metas,
            "tip": tip,
        }

