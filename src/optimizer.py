from tqdm import tqdm
from src.agent import GenerativeAgent
from ollama import chat

def run_one_simulation(you_traits: str, target_traits: str, scenario: str):
    you = GenerativeAgent("You", you_traits, goal="make target like me more + get IG")
    target = GenerativeAgent("Target", target_traits)
    
    history = []
    for turn in range(8):
        you_action = you.react(f"Turn {turn}: {history[-3:]}")
        target_action = target.react(f"You experienced: {you_action}")
        history.append(you_action)
        you.observe(target_action)
        target.observe(you_action)
    
    judge = chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": f"Rate liking increase 0-100. ONLY output the number.\nScenario: {scenario}\nLast turns:\n" + "\n".join(history[-4:])}],
        options={"num_predict": 30}
    )
    score = judge['message']['content']
    return {"path": history, "score": score}

def monte_carlo_optimize(scenario: str, you_traits: str, target_traits: str):
    results = []
    for i in tqdm(range(8), desc="Running universes"):
        results.append(run_one_simulation(you_traits, target_traits, scenario))
    best = max(results, key=lambda x: int(''.join(filter(str.isdigit, str(x['score']))[:3]) or 0))
    return best

