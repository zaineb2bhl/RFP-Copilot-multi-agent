"""SQLite persistence for runs and per-requirement compliance status (SQLAlchemy 2.0).

The LangGraph checkpointer stores the full graph state; this database holds the small,
queryable summary the API, UI and MCP server need.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import ForeignKey, String, Text, create_engine, delete, literal_column, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from rfp_copilot.models import ComplianceRow, Requirement

RunStatus = Literal["queued", "running", "awaiting_review", "completed", "rejected", "failed"]


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RunORM(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    rfp_filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    stage: Mapped[str | None] = mapped_column(String(64), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    output_path: Mapped[str | None] = mapped_column(Text, default=None)
    usage_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)


class RequirementORM(Base):
    __tablename__ = "requirements"

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    req_id: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(32))
    source_page: Mapped[int]
    status: Mapped[str] = mapped_column(String(32), default="pending")
    where_addressed: Mapped[str | None] = mapped_column(Text, default=None)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")


class RunRecord(BaseModel):
    """A run as exposed to the API/UI."""

    id: str
    rfp_filename: str
    status: RunStatus
    stage: str | None
    error: str | None
    output_path: str | None
    usage: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RequirementStatusRecord(BaseModel):
    """A requirement with its compliance status ("pending" until export)."""

    id: str
    text: str
    type: str
    source_page: int
    status: str
    where_addressed: str | None
    sources: list[str]


def _run_record(row: RunORM) -> RunRecord:
    return RunRecord(
        id=row.id,
        rfp_filename=row.rfp_filename,
        status=row.status,  # type: ignore[arg-type]
        stage=row.stage,
        error=row.error,
        output_path=row.output_path,
        usage=json.loads(row.usage_json or "{}"),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class Database:
    """Repository over the application database."""

    def __init__(self, url: str) -> None:
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._sessions()

    def create_run(self, run_id: str, rfp_filename: str) -> RunRecord:
        """Insert a new queued run."""
        with self._session() as session, session.begin():
            row = RunORM(id=run_id, rfp_filename=rfp_filename, status="queued")
            session.add(row)
        return _run_record(row)

    def update_run(self, run_id: str, **fields: Any) -> RunRecord:
        """Update columns of a run; `usage=` is stored as JSON."""
        with self._session() as session, session.begin():
            row = session.get(RunORM, run_id)
            if row is None:
                raise KeyError(run_id)
            if "usage" in fields:
                fields["usage_json"] = json.dumps(fields.pop("usage"))
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = _now()
        return _run_record(row)

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run, or None."""
        with self._session() as session:
            row = session.get(RunORM, run_id)
            return _run_record(row) if row else None

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """Most recent runs first."""
        with self._session() as session:
            # SQLite rowid breaks ties between runs created within the same clock tick.
            newest_first = (RunORM.created_at.desc(), literal_column("rowid").desc())
            rows = session.scalars(select(RunORM).order_by(*newest_first).limit(limit))
            return [_run_record(r) for r in rows]

    def replace_requirements(
        self,
        run_id: str,
        requirements: list[Requirement],
        compliance: list[ComplianceRow] | None = None,
    ) -> None:
        """Store the run's requirements, with compliance status when available."""
        by_id = {c.requirement_id: c for c in compliance or []}
        with self._session() as session, session.begin():
            session.execute(delete(RequirementORM).where(RequirementORM.run_id == run_id))
            for req in requirements:
                row = by_id.get(req.id)
                session.add(
                    RequirementORM(
                        run_id=run_id,
                        req_id=req.id,
                        text=req.text,
                        type=req.type,
                        source_page=req.source_page,
                        status=row.status if row else "pending",
                        where_addressed=row.where_addressed if row else None,
                        sources_json=json.dumps(row.sources if row else []),
                    )
                )

    def get_requirements(self, run_id: str) -> list[RequirementStatusRecord]:
        """Requirements of a run in id order."""
        with self._session() as session:
            rows = session.scalars(
                select(RequirementORM)
                .where(RequirementORM.run_id == run_id)
                .order_by(RequirementORM.req_id)
            )
            return [
                RequirementStatusRecord(
                    id=r.req_id,
                    text=r.text,
                    type=r.type,
                    source_page=r.source_page,
                    status=r.status,
                    where_addressed=r.where_addressed,
                    sources=json.loads(r.sources_json),
                )
                for r in rows
            ]
