"""SQLAlchemy database models for GuardRoute sessions and token usage auditing.

All models use the shared Base from common.models.database for unified
Alembic migration support across the monorepo.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Text, Float, JSON

from common.models.database import Base


class GuardRouteSession(Base):
    """Stores conversation metadata, complexity, run subagents, and final response."""

    __tablename__ = "guardroute_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    complexity = Column(String(50), nullable=True)
    subagents_ran = Column(JSON, nullable=True)  # List of subagents executed
    final_response = Column(Text, nullable=True)
    duration_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class GuardRouteUsage(Base):
    """Stores aggregated token usage details per session."""

    __tablename__ = "guardroute_usage"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
