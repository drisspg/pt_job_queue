from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"


class PtqError(Exception):
    pass


class JobNotFoundError(PtqError):
    pass


@dataclass
class JobRecord:
    job_id: str
    issue: int | None = None
    runs: int = 1
    agent: str = "claude"
    model: str = "opus"
    machine: str | None = None
    local: bool = False
    workspace: str = "~/ptq_workspace"
    pid: int | None = None
    initializing: bool = False
    pr_url: str | None = None
    human_note: str | None = None

    @property
    def target(self) -> str:
        return self.machine or "local"

    def to_dict(self) -> dict:
        d: dict = {
            "issue": self.issue,
            "runs": self.runs,
            "agent": self.agent,
            "model": self.model,
        }
        if self.local:
            d["local"] = True
        if self.machine:
            d["machine"] = self.machine
        d["workspace"] = self.workspace
        if self.pid is not None:
            d["pid"] = self.pid
        if self.initializing:
            d["initializing"] = True
        if self.pr_url:
            d["pr_url"] = self.pr_url
        if self.human_note:
            d["human_note"] = self.human_note
        return d

    @classmethod
    def from_dict(cls, job_id: str, data: dict) -> JobRecord:
        local = data.get("local", False)
        return cls(
            job_id=job_id,
            issue=data.get("issue"),
            runs=data.get("runs", 1),
            agent=data.get("agent", "claude"),
            model=data.get("model", "opus"),
            machine=data.get("machine"),
            local=local,
            workspace=data.get(
                "workspace", "~/.ptq_workspace" if local else "~/ptq_workspace"
            ),
            pid=data.get("pid"),
            initializing=data.get("initializing", False),
            pr_url=data.get("pr_url"),
            human_note=data.get("human_note"),
        )


@dataclass
class RunRequest:
    issue_data: dict | None = None
    issue_number: int | None = None
    message: str | None = None
    machine: str | None = None
    local: bool = False
    follow: bool = True
    model: str = "opus"
    max_turns: int = 100
    agent_type: str = "claude"
    existing_job_id: str | None = None
    verbose: bool = False


@dataclass
class PRResult:
    url: str
    branch: str
