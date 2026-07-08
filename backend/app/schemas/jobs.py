from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: dict
    error: Optional[str] = None
    done: bool
    result: Optional[Any] = None
