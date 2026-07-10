"""FastAPI application exposing the document pipeline."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import DocumentPipeline

app = FastAPI(
    title="doc-intelligence",
    description="Offline intelligent document processing API.",
    version=__version__,
)

# Built once at import time so requests are fast.
_pipeline: DocumentPipeline | None = None


def get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


class ParseRequest(BaseModel):
    text: str = Field(..., description="Raw document text to parse.")


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/parse")
def parse(req: ParseRequest) -> dict:
    return get_pipeline().process(req.text)
