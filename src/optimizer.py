import asyncio

from litellm import completion

from .agent import GenerativeAgent


async def run_one_simulation(
    you_traits: str,
    target_traits: str,
    scenario: str,
    num_turns: int = 8,
):
    you = GenerativeAgent("You", you_traits, goal="make target like me more + get IG")
    target = GenerativeAgent("Target", target_traits)

    history = []
    for turn in range(num_turns):
        you_action = you.react(f"Current situation after turn {turn}: {history}")
        target_action = target.react(f"You just experienced: {you_action}")
        history.append(f"t={turn*2}s: You: {you_action} | Target: {target_action}")
        you.observe(target_action)
        target.observe(you_action)

    judge_prompt = (
        "Rate final liking increase 0-100 and realism. "
        f"Scenario: {scenario}\nConversation:\n" + "\n".join(history)
    )
    score_response = completion(
        model="ollama/qwen2.5:14b",
        messages=[{"role": "user", "content": judge_prompt}],
    )
    score = score_response.choices[0].message.content
    return {"path": history, "score": score}


async def monte_carlo_optimize(
    scenario: str,
    you_traits: str,
    target_traits: str,
    num_simulations: int = 30,
):
    tasks = [run_one_simulation(you_traits, target_traits, scenario) for _ in range(num_simulations)]
    results = await asyncio.gather(*tasks)

    def _score_to_int(score_text: str) -> int:
        head = (score_text or "")[:3]
        digits = "".join(filter(str.isdigit, head))
        return int(digits or 0)

    best = max(results, key=lambda x: _score_to_int(x["score"]))
    return best

