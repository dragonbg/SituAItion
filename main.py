from fastapi import FastAPI
from src.optimizer import monte_carlo_optimize
import gradio as gr
import asyncio

app = FastAPI()

@app.post("/predict")
async def predict(scenario: str, you_traits: str, target_traits: str):
    result = await monte_carlo_optimize(scenario, you_traits, target_traits)
    return {"golden_path": result["path"], "success_estimate": result["score"]}

# Gradio UI for quick testing
def gradio_ui(scenario, you_traits, target_traits):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(monte_carlo_optimize(scenario, you_traits, target_traits))
    return "\n\n".join(result["path"]) + "\n\n" + result["score"]

iface = gr.Interface(
    fn=gradio_ui,
    inputs=[
        gr.Textbox(label="Scenario", placeholder="Coworker X, not close friends, want her to like me more and give IG"),
        gr.Textbox(label="Your traits", placeholder="28M Bulgarian, confident but subtle"),
        gr.Textbox(label="Target traits", placeholder="25F introvert, values genuine humor and light physical contact")
    ],
    outputs="text",
    title="SituAItion — Multiverse Social God",
    description="I ran thousands of parallel universes. Here is the one where you win."
)
iface.launch(share=False)