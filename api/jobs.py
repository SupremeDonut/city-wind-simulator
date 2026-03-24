import asyncio
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.RUNNING
    progress: float = 0.0          # 0–1
    result_path: str | None = None
    error: str | None = None
    # WebSocket queues — one per connected client
    subscribers: list[asyncio.Queue] = field(default_factory=list)


# Module-level store — lives for the lifetime of the process
store: dict[str, Job] = {}
