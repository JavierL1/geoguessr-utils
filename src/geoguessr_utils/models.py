from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Player(SQLModel, table=True):
    __tablename__ = "players"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

    # Relationships (back-references)
    matches_as_player_a: List["Match"] = Relationship(
        sa_relationship_kwargs={"primaryjoin": "Player.id==Match.player_a"}
    )
    matches_as_player_b: List["Match"] = Relationship(
        sa_relationship_kwargs={"primaryjoin": "Player.id==Match.player_b"}
    )


class Match(SQLModel, table=True):
    __tablename__ = "matches"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    player_a: int = Field(foreign_key="players.id", ondelete="CASCADE")
    player_b: int = Field(foreign_key="players.id", ondelete="CASCADE")

    # Relationships
    games: List["Game"] = Relationship(
        back_populates="match", sa_relationship_kwargs={"cascade": "all, delete"}
    )


class Game(SQLModel, table=True):
    __tablename__ = "games"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    type: str

    # sa_column_kwargs explicitly tells SQLAlchemy the physical SQLite column name
    index: int = Field(alias="_index", sa_column_kwargs={"name": "_index"})

    match_id: int = Field(foreign_key="matches.id", ondelete="CASCADE")

    # Relationships
    match: Optional[Match] = Relationship(back_populates="games")
    rounds: List["Round"] = Relationship(
        back_populates="game", sa_relationship_kwargs={"cascade": "all, delete"}
    )


class Round(SQLModel, table=True):
    __tablename__ = "rounds"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    index: int = Field(alias="_index", sa_column_kwargs={"name": "_index"})
    score_a: int
    score_b: int
    game_id: int = Field(foreign_key="games.id", ondelete="CASCADE")

    # Relationships
    game: Optional[Game] = Relationship(back_populates="rounds")

    # Add relationship to RoundResult
    results: List["RoundResult"] = Relationship(
        back_populates="round", sa_relationship_kwargs={"cascade": "all, delete"}
    )


class RoundResult(SQLModel, table=True):
    __tablename__ = "round_result"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="rounds.id", ondelete="CASCADE", unique=True)

    old_multi: float = Field(default=1.0)
    new_multi_a: float = Field(default=1.0)
    new_multi_b: float = Field(default=1.0)

    raw_damage_to_a: float = Field(default=0.0)
    raw_damage_to_b: float = Field(default=0.0)

    old_damage_to_a: float = Field(default=0.0)
    old_damage_to_b: float = Field(default=0.0)

    new_damage_to_a: float = Field(default=0.0)
    new_damage_to_b: float = Field(default=0.0)

    old_game_over: bool = Field(default=False)
    new_game_over: bool = Field(default=False)

    new_damage_taken_a: float = Field(default=0.0)
    new_damage_taken_b: float = Field(default=0.0)

    old_damage_taken_a: float = Field(default=0.0)
    old_damage_taken_b: float = Field(default=0.0)

    # Relationship back to Round
    round: Optional["Round"] = Relationship(back_populates="results")
