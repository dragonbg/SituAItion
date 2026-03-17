from ollama import chat
from pydantic import BaseModel
import datetime
import random
import textwrap

class Memory(BaseModel):
    observation: str
    timestamp: datetime.datetime
    importance: int = 5

class GenerativeAgent:
    def __init__(self, name: str, traits: str, goal: str = ""):
        self.name = name
        self.traits = traits
        self.goal = goal
        self.memory_stream: list[Memory] = []
        self.clock = 0.0

    def observe(self, observation: str):
        self.memory_stream.append(Memory(observation=observation, timestamp=datetime.datetime.now()))

    def react(self, situation: str) -> str:
        context = textwrap.shorten(
            "\n".join([m.observation for m in self.memory_stream[-3:]]),
            width=600, placeholder="..."
        )

        prompt = f"""You are {self.name} ({self.traits}).
Context (last 3 turns only): {context}
Current time: {self.clock:.1f}s
Situation: {situation}

EXACT FORMAT (keep it short!):
1. t=XX.Xs: [specific micro-action: touch, posture, eye contact, laugh etc.]
2. t=XX.Xs: ...

Exact words: "..."
Be novel and slightly weird if it helps rapport. No clichés."""

        resp = chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 180, "temperature": 0.9}
        )
        self.clock += random.uniform(1.6, 3.5)
        return resp['message']['content']

