"""
backend/app/jobs/models.py
============================
Internal data structures for the background job manager. Not Pydantic —
these live only inside the process; JobRecord is converted to a
JobStatusResponse (backend/app/schemas/jobs.py) at the API boundary.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class JobRecord:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: dict = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
