import gradio as gr
import os
os.environ["OLLAMA_NUM_PARALLEL"] = "1"

from src.optimizer import beam_search_simulation, evolutionary_search_and_render, monte_carlo_optimize
from src.psyche_hat import PsycheHat

hat = PsycheHat()

def _format_plan_block(label: str, plan: dict) -> str:
    steps = "\n".join([f"- {s}" for s in plan.get("steps", [])])
    exact_words = plan.get("exact_words", "")
    fallback = plan.get("fallback_if_no", "")
    score = plan.get("score", "")
    return (
        f"{label} (score: {score}/100)\n"
        f"Title: {plan.get('title','')}\n\n"
        f"Steps:\n{steps}\n\n"
        f'Exact words:\n"{exact_words}"\n\n'
        f'If they say no:\n"{fallback}"'
    ).strip()


def _format_sim(label: str, sim: dict) -> str:
    history = "\n".join(sim.get("history", []))
    score = sim.get("score", "")
    notes = sim.get("judge_notes", "")
    return f"{label} (score: {score}/100)\n\n{history}\n\nJudge notes: {notes}".strip()


def gradio_ui(
    scenario,
    goal,
    you_traits,
    target_traits,
    mode,
    turns,
    branch_factor,
    beam_width,
    evo_sims,
    evo_turns,
    use_hat,
    ab_test_hat,
    candidates,
    judges,
    progress=gr.Progress(),
):
    try:
        if mode == "Branching simulation (beam search)":
            result = beam_search_simulation(
                scenario=scenario,
                goal=goal,
                you_traits=you_traits,
                target_traits=target_traits,
                turns=int(turns),
                branch_factor=int(branch_factor),
                beam_width=int(beam_width),
                progress=progress,
            )
            best = _format_sim("Best branch", result["best"])
            alts = []
            for i, alt in enumerate(result.get("alternatives", []), start=1):
                alts.append(_format_sim(f"Alternative branch #{i}", alt))
            meta = result.get("meta", {})
            footer = (
                f"\n\nExpanded turns={meta.get('turns')} | branch_factor={meta.get('branch_factor')} | beam_width={meta.get('beam_width')}"
            )
            return "\n\n---\n\n".join([best] + alts) + footer

        if mode == "Evolutionary (phase1/2) + render winner":
            result = evolutionary_search_and_render(
                scenario=scenario,
                goal=goal,
                you_traits=you_traits,
                target_traits=target_traits,
                num_sims=int(evo_sims),
                turns=int(evo_turns),
                judge_top_k=8,
                progress=progress,
                hat=hat if use_hat else None,
                ab_test_hat=bool(ab_test_hat),
            )
            best_branch = result["best"]
            header = (
                f"Best branch (score: {best_branch.get('score')}/100)\n"
                f"Approach: {best_branch.get('approach','')}\n"
            )
            if result.get("meta", {}).get("hat_tip") and use_hat:
                header += f"{result['meta']['hat_tip']}\n"
            convo = "\n".join(best_branch.get("history", []))
            playbook = result.get("playbook", "")
            footer = (
                f"\n\n---\n\nMeta: phase1={result['meta'].get('phase1')} total_sims={result['meta'].get('total_sims')} turns={result['meta'].get('turns')}"
            )
            return header + "\n\n--- MESSAGE-LEVEL WINNER ---\n" + convo + "\n\n--- MICRO-TACTICS PLAYBOOK ---\n" + playbook + footer

        # Default: plan sampling + judging
        result = monte_carlo_optimize(
            scenario=scenario,
            you_traits=you_traits,
            target_traits=target_traits,
            goal=goal,
            candidates=int(candidates),
            judges=int(judges),
            progress=progress,
        )

        best = _format_plan_block("Best plan", result["best"])
        alts = []
        for i, alt in enumerate(result.get("alternatives", []), start=1):
            alts.append(_format_plan_block(f"Alternative #{i}", alt))

        meta = result.get("meta", {})
        footer = f"\n\nRan {meta.get('candidates','?')} candidates × ~{meta.get('judges','?')} judge samples total."
        return "\n\n---\n\n".join([best] + alts) + footer

    except Exception as e:
        return (
            "Couldn't run.\n\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            "If you're using Ollama as the backend:\n"
            "- Install the Python package: pip install ollama\n"
            "- Start Ollama\n"
            "- Pull a model: ollama pull qwen3:8b\n"
        )

iface = gr.Interface(
    fn=gradio_ui,
    inputs=[
        gr.Textbox(label="Scenario", placeholder="You and X are coworkers; you chat sometimes but aren't close yet."),
        gr.Textbox(label="Goal", placeholder="Ask for their Instagram (with an easy out, no pressure)."),
        gr.Textbox(label="Your traits", placeholder="Direct, friendly, a bit shy at first."),
        gr.Textbox(label="Target traits", placeholder="Introverted; values straightforwardness and respect."),
        gr.Dropdown(
            choices=[
                "Evolutionary (phase1/2) + render winner",
                "Branching simulation (beam search)",
                "Plan sampler (single-turn strategies)",
            ],
            value="Evolutionary (phase1/2) + render winner",
            label="Mode",
        ),
        gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Turns (simulation mode)"),
        gr.Slider(minimum=2, maximum=12, value=6, step=1, label="Branch factor (simulation mode)"),
        gr.Slider(minimum=2, maximum=24, value=12, step=1, label="Beam width (simulation mode)"),
        gr.Slider(minimum=16, maximum=256, value=64, step=16, label="Evolutionary sims (evo mode)"),
        gr.Slider(minimum=2, maximum=10, value=6, step=1, label="Turns per sim (evo mode)"),
        gr.Checkbox(value=True, label="Use PsycheHat memory (evo mode)"),
        gr.Checkbox(value=False, label="A/B test PsycheHat (every other sim uses hat approach)"),
        gr.Slider(minimum=4, maximum=30, value=12, step=1, label="Candidate plans"),
        gr.Slider(minimum=4, maximum=60, value=24, step=1, label="Judge samples (stability vs speed)"),
    ],
    outputs="text",
    title="SituAItion — Consent-first interaction planner",
    description="Evolutionary search (phase1 explore → phase2 branch winners) on short messages, then renders micro-tactics once for the winner."
)
iface.launch()