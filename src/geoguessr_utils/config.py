import json
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    FPS: float = 59.94006
    RESOLUTION: str = "1080p"
    KNOWN_PLAYERS: list[str] = Field(default_factory=list)

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
