"""FastAPI application: upload an RFP, follow the run, review drafts, download the DOCX.

Run with: `uvicorn rfp_copilot.api.app:app --port 8010`
"""

# No `from __future__ import annotations`: FastAPI must resolve the local `Service` alias.

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from rfp_copilot import __version__
from rfp_copilot.config import get_settings
from rfp_copilot.db.repository import RequirementStatusRecord, RunRecord
from rfp_copilot.pipeline import PipelineService, ReviewDecision

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@lru_cache(maxsize=1)
def default_service() -> PipelineService:
    """The process-wide pipeline service built from environment settings."""
    return PipelineService(get_settings())


def create_app(service: PipelineService | None = None) -> FastAPI:
    """Build the API. Pass a `service` to inject test doubles."""
    app = FastAPI(title="RFP Copilot API", version=__version__)

    def get_service() -> PipelineService:
        return service or default_service()

    Service = Annotated[PipelineService, Depends(get_service)]

    def require_run(svc: PipelineService, run_id: str) -> RunRecord:
        run = svc.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return run

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/runs", status_code=202)
    async def create_run(
        svc: Service, background: BackgroundTasks, file: Annotated[UploadFile, File()]
    ) -> RunRecord:
        """Upload an RFP PDF and start the pipeline in the background."""
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Please upload a .pdf file")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF larger than 25 MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=415, detail="File is not a PDF")
        run = svc.create_run(file.filename or "rfp.pdf", data)
        background.add_task(svc.execute, run.id)
        return run

    @app.get("/runs")
    def list_runs(svc: Service, limit: int = 50) -> list[RunRecord]:
        return svc.db.list_runs(limit)

    @app.get("/runs/{run_id}")
    def get_run(svc: Service, run_id: str) -> RunRecord:
        return require_run(svc, run_id)

    @app.get("/runs/{run_id}/requirements")
    def get_requirements(svc: Service, run_id: str) -> list[RequirementStatusRecord]:
        require_run(svc, run_id)
        return svc.db.get_requirements(run_id)

    @app.get("/runs/{run_id}/review")
    def get_review(svc: Service, run_id: str) -> dict[str, Any]:
        """Drafts, reviewer scores and retrieved sources for the run."""
        require_run(svc, run_id)
        return svc.review_payload(run_id)

    @app.post("/runs/{run_id}/review", status_code=202)
    def submit_review(
        svc: Service,
        background: BackgroundTasks,
        run_id: str,
        decision: ReviewDecision,
        wait: bool = False,
    ) -> RunRecord:
        """Approve (optionally with edited sections) or reject a run awaiting review.

        With `wait=true` the request returns once the run has finished: after approval only
        the DOCX export remains (no LLM call), so this takes about a second.
        """
        run = require_run(svc, run_id)
        if run.status != "awaiting_review":
            raise HTTPException(status_code=409, detail=f"Run is {run.status}, not awaiting review")
        run = svc.db.update_run(run_id, status="running", stage="human_review")
        if wait:
            return svc.resume(run_id, decision)
        background.add_task(svc.resume, run_id, decision)
        return run

    @app.get("/runs/{run_id}/download")
    def download(svc: Service, run_id: str) -> FileResponse:
        run = require_run(svc, run_id)
        if run.status != "completed" or not run.output_path or not Path(run.output_path).exists():
            raise HTTPException(status_code=404, detail="No DOCX available for this run yet")
        stem = Path(run.rfp_filename).stem
        return FileResponse(
            run.output_path,
            filename=f"{stem}-response.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    return app


app = create_app()
