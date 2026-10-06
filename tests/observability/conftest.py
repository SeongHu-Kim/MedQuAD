"""Observability tests reuse the synthetic API fakes from tests/api."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from api_fakes import make_client  # noqa: E402, F401  (fixture re-export)
