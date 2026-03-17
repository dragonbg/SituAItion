from litellm import completion
from pydantic import BaseModel

import datetime


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
        self.current_time = datetime.datetime.now()

    def observe(self, observation: str):
        self.memory_stream.append(
            Memory(
                observation=observation,
                timestamp=self.current_time,
            )
        )

    def reflect(self):
        if len(self.memory_stream) < 5:
            return
        recent = "\n".join([m.observation for m in self.memory_stream[-10:]])
        prompt = (
            f"You are {self.name} ({self.traits}, goal: {self.goal}). Reflect on recent events and "
            f"give 3-5 insights about your feelings/relationships:\n{recent}\nOutput only bullets."
        )
        response = completion(
            model="ollama/qwen2.5:14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        insights = response.choices[0].message.content
        self.observe(f"Reflection: {insights}")

    def react(self, situation: str) -> str:
        relevant = sorted(self.memory_stream, key=lambda m: m.importance, reverse=True)[:5]
        context = "\n".join([m.observation for m in relevant])
        prompt = f"""You are {self.name} ({self.traits}). 
Context: {context}
Situation: {situation}
Respond with extremely specific micro-actions: exact words, timing (e.g. t=3.2s), body language, tone. Be novel and realistic."""
        response = completion(
            model="ollama/qwen2.5:14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return response.choices[0].message.content

