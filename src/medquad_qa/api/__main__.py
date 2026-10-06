"""Run the API: ``python -m medquad_qa.api`` (binds MEDQUAD_BIND_HOST:MEDQUAD_API_PORT, default 127.0.0.1:8000)."""

from __future__ import annotations

import uvicorn

from medquad_qa.api.app import create_app
from medquad_qa.api.settings import ServiceSettings
from medquad_qa.observability.logs import configure_logging


def main() -> None:
    settings = ServiceSettings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.api_port,
        access_log=False,  # ServiceMiddleware writes one JSON access line per request
        log_config=None,  # keep the JSON handler installed above
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
