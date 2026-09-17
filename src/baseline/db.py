"""
Database connection for baseline pipeline.

Uses SQLModel for database operations.
"""

from sqlmodel import SQLModel, create_engine, Session
import os

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./data/experiment.db")
engine = create_engine(DB_URL, echo=False)


def get_session():
    with Session(engine) as session:
        yield session
