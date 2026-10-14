from typing import List

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import Session

from geoguessr_utils.io.database import engine
from geoguessr_utils.models import RoundResult


def upsert_round_results(results: List[RoundResult]) -> None:
    """Upserts a list of RoundResult instances based on unique round_id."""
    if not results:
        return

    values_to_insert = [res.model_dump(exclude={"id"}) for res in results]

    stmt = sqlite_insert(RoundResult).values(values_to_insert)

    update_dict = {
        "old_multi": stmt.excluded.old_multi,
        "new_multi_a": stmt.excluded.new_multi_a,
        "new_multi_b": stmt.excluded.new_multi_b,
        "raw_damage_to_a": stmt.excluded.raw_damage_to_a,
        "raw_damage_to_b": stmt.excluded.raw_damage_to_b,
        "old_damage_to_a": stmt.excluded.old_damage_to_a,
        "old_damage_to_b": stmt.excluded.old_damage_to_b,
        "new_damage_to_a": stmt.excluded.new_damage_to_a,
        "new_damage_to_b": stmt.excluded.new_damage_to_b,
        "old_game_over": stmt.excluded.old_game_over,
        "new_game_over": stmt.excluded.new_game_over,
        "new_damage_taken_a": stmt.excluded.new_damage_taken_a,
        "new_damage_taken_b": stmt.excluded.new_damage_taken_b,
        "old_damage_taken_a": stmt.excluded.old_damage_taken_a,
        "old_damage_taken_b": stmt.excluded.old_damage_taken_b,
    }

    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["round_id"],
        set_=update_dict,
    )

    with Session(engine) as session:
        session.exec(upsert_stmt)
        session.commit()
