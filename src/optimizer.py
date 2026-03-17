from ollama import chat
from tqdm import tqdm

from src.agent import GenerativeAgent


def run_one_simulation(you_traits: str, target_traits: str, scenario: str, num_turns: int = 8):
    you = GenerativeAgent("You", you_traits, goal="make target like me more + get IG")
    target = GenerativeAgent("Target", target_traits)

    history: list[str] = []
    for turn in range(num_turns):
        you_action = you.react(f"Turn {turn+1}/{num_turns}. Recent: {history[-3:]}")
        target_action = target.react(f"You experienced: {you_action}")
        history.append(you_action)
        you.observe(target_action)
        target.observe(you_action)

    judge_msg = (
        "Rate liking increase 0-100. ONLY the number.\n"
        f"Scenario: {scenario}\n"
        "Last turns:\n"
        + "\n".join(history[-4:])
    )
    judge = chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": judge_msg}],
        options={"num_predict": 30},
    )
    score = judge["message"]["content"]
    return {"path": history, "score": score}


def monte_carlo_optimize(scenario: str, you_traits: str, target_traits: str, num_sims: int = 8):
    results = []
    for _ in tqdm(range(num_sims), desc="Universes"):
        results.append(run_one_simulation(you_traits, target_traits, scenario))
    best = max(results, key=lambda x: int("".join(filter(str.isdigit, str(x["score"]))[:3]) or 0))
    return best

