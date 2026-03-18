import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env_from_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


_load_env_from_file(ROOT / ".env")

# Longer timeout for heavy benchmark runs (default is 90s in LlmConfig).
os.environ.setdefault("SITUAITION_TIMEOUT_S", "600")

from src.optimizer import evolutionary_search_and_render

SCENARIO = "After-work rooftop bar meetup with a coworker — relaxed vibe, light music."
GOAL = "Escalate from casual chat to asking for Instagram with a clear out."
YOU_TRAITS = "Warm, witty, direct once comfortable."
TARGET_TRAITS = "Introverted, likes specificity, wary of try-hard energy."

BASE_KWARGS = dict(
    scenario=SCENARIO,
    goal=GOAL,
    you_traits=YOU_TRAITS,
    target_traits=TARGET_TRAITS,
    num_sims=32,
    turns=4,
    judge_top_k=8,
)


def run_case(*, use_agents: bool) -> tuple[float, dict]:
    random.seed(1337)
    start = time.perf_counter()
    result = evolutionary_search_and_render(
        **BASE_KWARGS,
        use_generative_agents=use_agents,
    )
    elapsed = time.perf_counter() - start
    return elapsed, result.get("meta", {})


def main():
    cases = [(False, "GA disabled"), (True, "GA enabled")]
    rows: list[tuple[str, float, dict]] = []
    for flag, label in cases:
        print(f"Running benchmark: {label}...", flush=True)
        elapsed, meta = run_case(use_agents=flag)
        rows.append((label, elapsed, meta))
        print(f"{label}: {elapsed:.2f}s (meta={meta})", flush=True)

    print("\nSummary:")
    for label, elapsed, meta in rows:
        phase1 = meta.get("phase1")
        total_sims = meta.get("total_sims")
        turns = meta.get("turns")
        print(
            f"- {label}: {elapsed:.2f}s | phase1={phase1} | total_sims={total_sims} | turns={turns}",
            flush=True,
        )


if __name__ == "__main__":
    main()
