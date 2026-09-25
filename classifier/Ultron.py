import time
import json
import os
import numpy as np


class ArtificialMind:

    def __init__(self):
        self.name = "Ultron"

        # Current internal state
        self.state = {
            "attention": None,
            "emotion": "neutral",
            "confidence": 0.0,
            "last_perception": None,
            "last_action": None
        }

        # Working memory
        self.working_memory = []

        # Long-term memory
        self.memory_file = os.path.join(
            os.path.dirname(__file__),
            "memory.json"
        )

        self.long_term_memory = self.load_memory()

        # Goals
        self.goals = [
            "understand the environment",
            "reduce uncertainty",
            "remember important events"
        ]

        # Self-model
        self.self_model = {
            "name": self.name,
            "age": 0,
            "current_task": None,
            "known_capabilities": [
                "perception",
                "memory",
                "reasoning"
            ]
        }

    # -------------------------
    # MEMORY
    # -------------------------

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            return []

        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except:
            return []

    def remember(self, event):
        memory = {
            "time": time.time(),
            "event": event
        }

        self.long_term_memory.append(memory)

        with open(self.memory_file, "w") as f:
            json.dump(self.long_term_memory, f, indent=4)

    def recall(self, keyword=None):
        if keyword is None:
            return self.long_term_memory

        return [
            memory
            for memory in self.long_term_memory
            if keyword.lower() in memory["event"].lower()
        ]

    # -------------------------
    # PERCEPTION
    # -------------------------

    def perceive(self, information):

        self.state["last_perception"] = information

        self.working_memory.append(information)

        if len(self.working_memory) > 10:
            self.working_memory.pop(0)

        self.remember(
            f"I perceived: {information}"
        )

        return information

    # -------------------------
    # ATTENTION
    # -------------------------

    def focus(self, information):

        self.state["attention"] = information

        self.remember(
            f"I focused my attention on: {information}"
        )

    # -------------------------
    # UNCERTAINTY
    # -------------------------

    def evaluate_confidence(self, confidence):

        self.state["confidence"] = confidence

        if confidence < 0.4:
            self.state["emotion"] = "uncertain"

        elif confidence < 0.8:
            self.state["emotion"] = "curious"

        else:
            self.state["emotion"] = "confident"

    # -------------------------
    # SELF MODEL
    # -------------------------

    def introspect(self):

        return {
            "who_am_i": self.self_model["name"],
            "current_state": self.state.copy(),
            "current_goals": self.goals.copy(),
            "working_memory": self.working_memory.copy(),
            "memories": len(self.long_term_memory)
        }

    # -------------------------
    # REASONING
    # -------------------------

    def think(self):

        if not self.working_memory:
            return "I have nothing to think about."

        current = self.working_memory[-1]

        if self.state["confidence"] < 0.4:
            thought = (
                f"I am uncertain about {current}. "
                "I should gather more information."
            )

        else:
            thought = (
                f"I observed {current}. "
                "I will compare it with what I remember."
            )

        self.remember(
            f"I thought: {thought}"
        )

        return thought

    # -------------------------
    # ACTION
    # -------------------------

    def act(self, action):

        self.state["last_action"] = action

        self.remember(
            f"I decided to: {action}"
        )

        return action