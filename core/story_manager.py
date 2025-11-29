from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from core.spell_engine import SpellType


class StoryStepType(Enum):
    """Different phases of the guided story."""

    EXPLANATION = "explanation"
    PRACTICE = "practice"


@dataclass
class StoryStep:
    id: int
    title: str
    description: str
    step_type: StoryStepType
    required_spell: Optional[SpellType]
    success_message: Optional[str] = None
    next_step_id: Optional[int] = None


class StoryManager:
    def __init__(self):
        self.current_step_id = 1
        self.steps: Dict[int, StoryStep] = self._init_steps()

    def _init_steps(self) -> Dict[int, StoryStep]:
        return {
            # Lumos briefing + practice
            1: StoryStep(
                id=1,
                title="Lumos Primer",
                description=(
                    "You stand in a corridor swallowed by darkness. The first lesson is simple: "
                    "focus on the tip of your wand, picture a warm glow, and remember that even a single "
                    "spark can chase shadows away. This is Lumos."
                ),
                step_type=StoryStepType.EXPLANATION,
                required_spell=SpellType.LUMOS,
                next_step_id=2,
            ),
            2: StoryStep(
                id=2,
                title="Practice Lumos",
                description=(
                    "Time to put theory into action. Take a steady breath and speak 'Lumos' to light the way."
                ),
                step_type=StoryStepType.PRACTICE,
                required_spell=SpellType.LUMOS,
                success_message="Brilliant! The corridor bursts with light and a solid oak door appears ahead.",
                next_step_id=3,
            ),
            # Wingardium Leviosa briefing + practice
            3: StoryStep(
                id=3,
                title="Levitation Lesson",
                description=(
                    "Just outside, a pile of stones blocks your path. Levitation is about balance—controlled movement, "
                    "a graceful swish-and-flick, and a confident voice guiding the magic."
                ),
                step_type=StoryStepType.EXPLANATION,
                required_spell=SpellType.WINGARDIUM_LEVIOSA,
                next_step_id=4,
            ),
            4: StoryStep(
                id=4,
                title="Practice Wingardium Leviosa",
                description="Lift the stones high using 'Wingardium Leviosa' and clear the courtyard.",
                step_type=StoryStepType.PRACTICE,
                required_spell=SpellType.WINGARDIUM_LEVIOSA,
                success_message="The rocks float aside gracefully. The courtyard opens up—training complete!",
                next_step_id=5,
            ),
            # Cursor briefing + practice
            5: StoryStep(
                id=5,
                title="Cursor Conjuring",
                description=(
                    "In the Great Hall, you discover an ancient spell that brings drawings to life. "
                    "The Cursor spell allows you to draw a pattern in the air, and magic will identify "
                    "what you've created—a ball, a cat, a heart, a pizza, a star, or a wand. "
                    "Draw clearly and let your imagination guide your wand."
                ),
                step_type=StoryStepType.EXPLANATION,
                required_spell=SpellType.CURSOR,
                next_step_id=6,
            ),
            6: StoryStep(
                id=6,
                title="Practice Cursor",
                description=(
                    "Cast 'Cursor' and draw a pattern in the air with your wand. "
                    "You have 5 seconds to complete your drawing. The magic will identify what you've created!"
                ),
                step_type=StoryStepType.PRACTICE,
                required_spell=SpellType.CURSOR,
                success_message="Magnificent! Your drawing has been recognized and brought to life as a 3D object!",
                next_step_id=None,
            ),
        }

    def get_current_step(self) -> Optional[StoryStep]:
        return self.steps.get(self.current_step_id)

    def advance_step(self) -> bool:
        """Advance to the next step. Returns True if there is a next step."""
        current = self.get_current_step()
        if current and current.next_step_id:
            self.current_step_id = current.next_step_id
            return True
        return False

    def reset(self):
        self.current_step_id = 1

