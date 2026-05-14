"""Rule-based final goal / landmark phrase extraction for R2R-CE assets."""

from __future__ import annotations

import re

from platonicnav.schemas import InstructionGoalExtraction


_STOP_WORDS = {
    "the",
    "a",
    "an",
    "there",
    "here",
    "it",
    "area",
    "room",
}


def _clean_phrase(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ -]+", " ", text).lower()
    words = [w for w in text.split() if w and w not in _STOP_WORDS]
    return " ".join(words).strip()


class RuleBasedGoalExtractor:
    """Extract a final noun/landmark phrase from route instructions.

    This is intentionally conservative and transparent for the public asset
    repo. Production experiments can replace it with a stronger parser while
    keeping the same output schema.
    """

    def __init__(self, *, fallback_last_words: int = 4) -> None:
        self.fallback_last_words = int(fallback_last_words)
        self.patterns = [
            re.compile(r"(?:stop|wait|end|finish)\s+(?:at|by|near|beside|next to|in front of)\s+(?:the\s+|a\s+|an\s+)?([^.;,]+)", re.I),
            re.compile(r"(?:go|walk|move|head)\s+(?:to|toward|towards)\s+(?:the\s+|a\s+|an\s+)?([^.;,]+)", re.I),
            re.compile(r"(?:by|near|beside|next to|in front of)\s+(?:the\s+|a\s+|an\s+)?([^.;,]+)$", re.I),
        ]

    def extract(self, instruction: str) -> InstructionGoalExtraction:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must be non-empty")
        for pattern in self.patterns:
            matches = list(pattern.finditer(instruction))
            if matches:
                phrase = _clean_phrase(matches[-1].group(1))
                if phrase:
                    return InstructionGoalExtraction(
                        instruction=instruction,
                        extracted_goal=phrase,
                        method="rule_based_final_goal_phrase",
                        diagnostics={"pattern": pattern.pattern},
                    )
        words = _clean_phrase(instruction).split()
        phrase = " ".join(words[-self.fallback_last_words :]).strip()
        if not phrase:
            raise ValueError("could not extract goal phrase from instruction")
        return InstructionGoalExtraction(
            instruction=instruction,
            extracted_goal=phrase,
            method="rule_based_last_words_fallback",
            diagnostics={"fallback_last_words": self.fallback_last_words},
        )
