import sys
from itertools import accumulate
from math import floor
from typing import List

from sqlmodel import Session

from geoguessr_utils.io.database import engine
from geoguessr_utils.io.repository import upsert_round_results
from geoguessr_utils.models import Game, Round, RoundResult

OLD_MAX_HEALTH_POINTS = 6000
OLD_MAX_ROUNDS = 10
NEW_MAX_HEALTH_POINTS = 5000


def get_game_with_rounds(game_id: int) -> tuple[Game | None, List[Round]]:
    """Alternative approach: Query Game object and access .rounds relationship."""
    with Session(engine) as session:
        game = session.get(Game, game_id)
        if not game:
            return None, []
        # Accessing relationship directly (sorted in memory if needed)
        rounds = sorted(game.rounds, key=lambda r: r.index)
        return game, rounds


def resolve_round(
    previous_round_result: RoundResult, current_round: Round
) -> RoundResult:
    raw_damage_to_a = 0
    raw_damage_to_b = 0
    old_damage_to_a = 0
    old_damage_to_b = 0
    new_damage_to_a = 0
    new_damage_to_b = 0

    tied = current_round.score_a == current_round.score_b

    if not tied:
        raw_damage_to_a = (
            current_round.score_b - current_round.score_a
            if current_round.score_b > current_round.score_a
            else 0
        )
        raw_damage_to_b = (
            current_round.score_a - current_round.score_b
            if current_round.score_a > current_round.score_b
            else 0
        )

    old_multi = previous_round_result.old_multi + 0.5

    new_multi_a = previous_round_result.new_multi_a
    new_multi_b = previous_round_result.new_multi_b

    if tied or raw_damage_to_b > 0:
        new_multi_a = previous_round_result.new_multi_a + 0.5
        old_damage_to_b = floor(raw_damage_to_b * previous_round_result.old_multi)
        new_damage_to_b = floor(raw_damage_to_b * previous_round_result.new_multi_a)

    if tied or raw_damage_to_a > 0:
        new_multi_b = previous_round_result.new_multi_b + 0.5
        old_damage_to_a = floor(raw_damage_to_a * previous_round_result.old_multi)
        new_damage_to_a = floor(raw_damage_to_a * previous_round_result.new_multi_b)

    new_damage_taken_a = previous_round_result.new_damage_taken_a + new_damage_to_a
    new_damage_taken_b = previous_round_result.new_damage_taken_b + new_damage_to_b

    old_damage_taken_a = previous_round_result.old_damage_taken_a + old_damage_to_a
    old_damage_taken_b = previous_round_result.old_damage_taken_b + old_damage_to_b

    old_game_over = (
        old_damage_taken_a > OLD_MAX_HEALTH_POINTS
        or old_damage_taken_b > OLD_MAX_HEALTH_POINTS
        or (current_round.index + 1) >= OLD_MAX_ROUNDS
    )
    new_game_over = (
        new_damage_taken_a > NEW_MAX_HEALTH_POINTS
        or new_damage_taken_b > NEW_MAX_HEALTH_POINTS
    )

    return RoundResult(
        round_id=current_round.id,
        raw_damage_to_a=raw_damage_to_a,
        raw_damage_to_b=raw_damage_to_b,
        old_multi=old_multi,
        new_multi_a=new_multi_a,
        new_multi_b=new_multi_b,
        old_damage_to_a=old_damage_to_a,
        old_damage_to_b=old_damage_to_b,
        new_damage_to_a=new_damage_to_a,
        new_damage_to_b=new_damage_to_b,
        new_damage_taken_a=new_damage_taken_a,
        new_damage_taken_b=new_damage_taken_b,
        old_damage_taken_a=old_damage_taken_a,
        old_damage_taken_b=old_damage_taken_b,
        old_game_over=old_game_over,
        new_game_over=new_game_over,
    )


def resolve_rounds(rounds: List[Round]) -> List[RoundResult]:
    return list(accumulate(rounds, resolve_round, initial=RoundResult()))[1:]


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python -m geoguessr_utils.get_game_rounds <game_id>")
        sys.exit(1)

    try:
        target_game_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: Game ID must be an integer. Got '{sys.argv[1]}'")
        sys.exit(1)

    game, rounds = get_game_with_rounds(target_game_id)

    round_results = resolve_rounds(rounds)
    upsert_round_results(round_results)

    if not game:
        print(f"Game with ID {target_game_id} not found.")
        return

    print(
        f"\n--- Game ID: {game.id} (Type: {game.type}, Match ID: {game.match_id}) ---"
    )
    if not rounds:
        print("No rounds found for this game.")
        return

    print(f"{'Round Index':<12} | {'Score A':<10} | {'Score B':<10}")
    print("-" * 40)
    for r in rounds:
        print(f"{r.index:<12} | {r.score_a:<10} | {r.score_b:<10}")


if __name__ == "__main__":
    main()
