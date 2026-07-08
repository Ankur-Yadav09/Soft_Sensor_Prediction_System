"""
backend/app/jobs/manager.py
==============================
In-process background job runner for long-running operations (model
training, feature selection). Single-user, single-process today: a
ThreadPoolExecutor plus an in-memory dict of job_id -> JobRecord, guarded by
a lock. If this ever needs to run across multiple processes/workers, swap
the dict for Redis (or similar) behind the same submit()/get() interface —
no caller-facing change required.

This is a short-lived, ID-addressed cache of in-flight work, not persisted
application state: whatever a job produces (e.g. a trained model) is
expected to be written to disk/DB by the wrapped function itself (via the
existing src.persistence.model_store / src.data.database calls) before the
job is marked done, so the JobRecord can be safely discarded afterward.

Public API
----------
job_manager.submit(fn, *args, **kwargs) -> job_id
job_manager.get(job_id) -> JobRecord | None
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Literal, Optional

from backend.app.jobs.models import JobRecord, JobStatus

# Which progress-callback shape the wrapped function actually accepts:
#   "epoch"   — progress_callback(current, total) + status_callback(message)
#               (src.training.trainer.train_model, src.training.train_lstm.train_lstm)
#   "message" — progress_callback(message) only, no status_callback
#               (src.feature_selection.auto_selector.run_auto_feature_selection /
#                run_per_target_auto_selection)
#   "none"    — no callback kwargs at all. Required for train_sklearn_model /
#               train_kalman_model, which absorb arbitrary kwargs via **hparams
#               and would forward a stray callback straight into an estimator
#               constructor if one were passed.
ProgressMode = Literal["epoch", "message", "none"]


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        fn: Callable,
        *args,
        progress_mode: ProgressMode = "epoch",
        **kwargs,
    ) -> str:
        """Run ``fn`` in a background thread and return a job_id immediately."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobRecord(id=job_id)

        def _progress_cb(current: int, total: int) -> None:
            with self._lock:
                self._jobs[job_id].progress = {
                    **self._jobs[job_id].progress,
                    "current": current,
                    "total": total,
                }

        def _status_cb(message: str) -> None:
            with self._lock:
                self._jobs[job_id].progress = {
                    **self._jobs[job_id].progress,
                    "message": message,
                }

        def _run() -> None:
            with self._lock:
                self._jobs[job_id].status = JobStatus.RUNNING
            try:
                if progress_mode == "epoch":
                    result = fn(
                        *args,
                        progress_callback=_progress_cb,
                        status_callback=_status_cb,
                        **kwargs,
                    )
                elif progress_mode == "message":
                    result = fn(*args, progress_callback=_status_cb, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                with self._lock:
                    self._jobs[job_id].result = result
                    self._jobs[job_id].status = JobStatus.DONE
            except Exception as exc:  # noqa: BLE001 — surfaced via job status, not raised
                with self._lock:
                    self._jobs[job_id].error = str(exc)
                    self._jobs[job_id].status = JobStatus.ERROR

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)


# Process-wide singleton — legitimate here since jobs are addressed by their
# own id, not treated as "the current" anything (see backend/app/jobs/manager.py docstring).
job_manager = JobManager()
