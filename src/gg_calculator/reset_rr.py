from sqlmodel import SQLModel
from gg_calculator.database import engine
from gg_calculator.models import RoundResult


def reset_round_results_table():
    # Drops the old table without the UNIQUE constraint
    RoundResult.__table__.drop(engine, checkfirst=True)

    # Recreates all tables based on updated models
    SQLModel.metadata.create_all(engine)
    print("Updated round_result table with UNIQUE constraint.")


if __name__ == "__main__":
    reset_round_results_table()
