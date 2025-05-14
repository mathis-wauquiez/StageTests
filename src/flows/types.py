
# -----------------------------------------------------------------------------
# This file contains the types and configuration for the flow module.
# ------------------------------------------------------------------------------


from enum import Enum
from typing import Optional, Dict, Any, Generator, Literal, Tuple
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------

class Predicts(str, Enum):
    X0 = "x_0"
    X1 = "x_1"
    SCORE = "score"
    VELOCITY = "v"

    @classmethod
    def from_any(cls, value: "Predicts | str") -> "Predicts":
        if isinstance(value, cls):
            return value
        key = str(value).lower()
        if key not in cls._ALIASES:
            raise ValueError(f"Unknown Predicts value: {value}")
        return cls._ALIASES[key]

Predicts._ALIASES = {
    "x0":       Predicts.X0,
    "x_0":      Predicts.X0,
    "x1":       Predicts.X1,
    "x_1":      Predicts.X1,
    "score":    Predicts.SCORE,
    "velocity": Predicts.VELOCITY,
    "v":        Predicts.VELOCITY,
}

class Guidance(str, Enum):
    NONE = "none"
    CFG = "CFG"
    CLASSIFIER = "classifier"


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------

@dataclass
class FlowConfig:
    predicts: Predicts = Predicts.X1
    guidance: Guidance = Guidance.NONE
    guidance_scale: float = 1.0
    guided_prob: float = 0.9 # train 90% of the time with labels
    compile: bool = False

    def __post_init__(self):
        # Ensure string aliases work when loaded from YAML
        if not isinstance(self.predicts, Predicts):
            self.predicts = Predicts.from_any(self.predicts)
        if not isinstance(self.guidance, Guidance):
            self.guidance = Guidance(self.guidance)