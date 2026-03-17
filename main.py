import gradio as gr
import os
os.environ["OLLAMA_NUM_PARALLEL"] = "1"

from src.optimizer import monte_carlo_optimize

def gradio_ui(scenario, you_traits, target_traits):
    result = monte_carlo_optimize(scenario, you_traits, target_traits)
    return "\n\n".join(result["path"]) + f"\n\n🎯 Success estimate: {result['score']}"

iface = gr.Interface(
    fn=gradio_ui,
    inputs=[
        gr.Textbox(label="Scenario", placeholder="Coworker X, not close friends, want her to like me more and give IG"),
        gr.Textbox(label="Your traits", placeholder="28M Bulgarian, confident but subtle"),
        gr.Textbox(label="Target traits", placeholder="25F introvert, values genuine humor and light physical contact")
    ],
    outputs="text",
    title="SituAItion — Multiverse Social God",
    description="~40-60s total on 1070 Ti (precise timing + novel micro-actions)"
)
iface.launch()