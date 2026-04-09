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


class RebaseState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    NEEDS_HUMAN = "needs_human"
    FAILED = "failed"


@dataclass
class RebaseInfo:
    state: RebaseState = RebaseState.IDLE
    target_ref: str = ""
    before_sha: str = ""
    after_sha: str = ""
    attempts: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        if self.state == RebaseState.IDLE:
            return {}
        d: dict = {"state": self.state.value}
        if self.target_ref:
            d["target_ref"] = self.target_ref
        if self.before_sha:
            d["before_sha"] = self.before_sha
        if self.after_sha:
            d["after_sha"] = self.after_sha
        if self.attempts:
            d["attempts"] = self.attempts
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RebaseInfo:
        if not data:
            return cls()
        return cls(
            state=RebaseState(data.get("state", "idle")),
            target_ref=data.get("target_ref", ""),
            before_sha=data.get("before_sha", ""),
            after_sha=data.get("after_sha", ""),
            attempts=data.get("attempts", 0),
            error=data.get("error", ""),
        )


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
    pr_title: str | None = None
    rebase: RebaseInfo | None = None
    name: str | None = None
    repo: str = "pytorch"

    @property
    def target(self) -> str:
        return self.machine or "local"

    @property
    def rebase_info(self) -> RebaseInfo:
        if self.rebase is None:
            self.rebase = RebaseInfo()
        return self.rebase

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
        if self.pr_title:
            d["pr_title"] = self.pr_title
        if self.name:
            d["name"] = self.name
        if self.repo != "pytorch":
            d["repo"] = self.repo
        if self.rebase is not None:
            rebase_data = self.rebase.to_dict()
            if rebase_data:
                d["rebase"] = rebase_data
        return d

    @classmethod
    def from_dict(cls, job_id: str, data: dict) -> JobRecord:
        local = data.get("local", False)
        rebase_data = data.get("rebase")
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
            pr_title=data.get("pr_title"),
            rebase=RebaseInfo.from_dict(rebase_data) if rebase_data else None,
            name=data.get("name"),
            repo=data.get("repo", "pytorch"),
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
    name: str | None = None
    repo: str = "pytorch"


@dataclass
class PRResult:
    url: str
    branch: str
