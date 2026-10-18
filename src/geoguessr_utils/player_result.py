from typing import TypedDict

from geoguessr_utils.parsers import parsing


class PlayerResult(TypedDict):
    match_score: int | None
    username: str | None
    round_score: int | None
    distance: parsing.Distance | None
    health_points: int | None
