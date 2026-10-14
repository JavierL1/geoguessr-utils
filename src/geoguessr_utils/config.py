import json
from pathlib import Path

from pydantic import BaseModel, Field

ROI_NAMES: list[str] = [
    "discover_left",
    "discover_right",
    "discover_center",
    "overall_score_left",
    "overall_score_right",
    "round_score_left",
    "round_score_right",
    "distance_left",
    "distance_right",
    "health_left",
    "health_right",
    "username_left",
    "username_right",
]


class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int


class AppConfig(BaseModel):
    FPS: float = 59.94006
    KNOWN_PLAYERS: list[str] = Field(default_factory=list)
    ROIS: dict[str, ROI] = Field(default_factory=dict)

    @classmethod
    def load(cls, filepath: Path) -> "AppConfig":
        if not filepath.exists():
            return cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, filepath: Path):
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    def missing_rois(self) -> list[str]:
        return [name for name in ROI_NAMES if name not in self.ROIS]
