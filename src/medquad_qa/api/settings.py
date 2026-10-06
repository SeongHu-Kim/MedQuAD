"""Service settings, read from ``MEDQUAD_*`` environment variables (see .env.example).

Owner: service-platform-engineer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_VERSION = "0.1.0"


class ServiceSettings(BaseSettings):
    """API process settings. Pipeline/model/index settings belong to their owners' factories."""

    model_config = SettingsConfigDict(env_prefix="MEDQUAD_", extra="ignore")

    bind_host: str = Field(default="127.0.0.1", description="Interface uvicorn binds to. Keep loopback by default.")
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_body_bytes: int = Field(default=16_384, ge=1024, le=1_048_576, description="Request bodies above -> 413.")
    request_timeout_s: float = Field(
        default=120.0, gt=0, description="API backstop around pipeline.answer; the generator has its own 60 s budget."
    )
    queue_timeout_s: float = Field(
        default=10.0, ge=0, description="Max wait for a generation slot before replying 503 busy."
    )
    max_concurrent_generations: int = Field(default=1, ge=1, le=8, description="One global generation semaphore.")
    retry_after_s: int = Field(default=5, ge=1, description="Retry-After header value on 503 responses.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_content: bool = Field(
        default=False,
        description="Opt-in: log sha256 + first 200 chars of question/answer. Off by default (no raw medical text).",
    )
    offline_eval_path: str | None = Field(
        default=None, description="JSON summary from the evaluator, exported as medquad_offline_eval_* gauges."
    )
    build_in_background: bool = Field(
        default=True, description="Build the pipeline after startup so /health/live answers immediately."
    )
