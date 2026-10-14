from sqlmodel import Session, SQLModel, create_engine

# SQLite file placed in root or local path
DATABASE_URL = "sqlite:///geoguessr_data.db"

# connect_args={"check_same_thread": False} allows SQLite to handle
# concurrent requests across threads safely in local dev
engine = create_engine(
    DATABASE_URL, echo=True, connect_args={"check_same_thread": False}
)


def init_db():
    """Create tables if they don't already exist."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Yield a database session context."""
    with Session(engine) as session:
        yield session
