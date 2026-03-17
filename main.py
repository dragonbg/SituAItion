import os

os.environ["OLLAMA_NUM_PARALLEL"] = "1"

from fastapi import FastAPI
from src.optimizer import monte_carlo_optimize
import gradio as gr

app = FastAPI()

@app.post("/predict")
def predict(scenario: str, you_traits: str, target_traits: str):
    result = monte_carlo_optimize(scenario, you_traits, target_traits)
    return {"golden_path": result["path"], "success_estimate": result["score"]}

# Gradio UI for quick testing
def gradio_ui(scenario, you_traits, target_traits):
    result = monte_carlo_optimize(scenario, you_traits, target_traits, num_sims=8)
    return "\n\n".join(result["path"]) + "\n\nSuccess estimate: " + str(result["score"])

iface = gr.Interface(
    fn=gradio_ui,
    inputs=[
        gr.Textbox(label="Scenario", placeholder="Coworker X, not close friends, want her to like me more and give IG"),
        gr.Textbox(label="Your traits", placeholder="28M Bulgarian, confident but subtle"),
        gr.Textbox(label="Target traits", placeholder="25F introvert, values genuine humor and light physical contact")
    ],
    outputs="text",
    title="SituAItion — Multiverse Social God",
    description="Pure Ollama + tight caps for faster runs."
)
iface.launch(share=False)