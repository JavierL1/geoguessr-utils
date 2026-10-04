from typing import List
from sqlmodel import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from gg_calculator.database import engine
from gg_calculator.models import RoundResult


def upsert_round_results(results: List[RoundResult]) -> None:
    """Upserts a list of RoundResult instances based on unique round_id."""
    if not results:
        return

    # Convert SQLModel instances into lists of raw dictionaries
    values_to_insert = [res.model_dump(exclude={"id"}) for res in results]

    # Build SQLite INSERT statement
    stmt = sqlite_insert(RoundResult).values(values_to_insert)

    # Columns to update if round_id conflicts
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

    # Add ON CONFLICT DO UPDATE clause
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["round_id"],
        set_=update_dict,
    )

    with Session(engine) as session:
        session.exec(upsert_stmt)
        session.commit()
