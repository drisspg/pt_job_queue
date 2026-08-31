from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


class SubmissionMode(Enum):
    PULL_REQUEST = "pull_request"
    GHSTACK = "ghstack"


@dataclass
class RebaseInfo:
    state: RebaseState = RebaseState.IDLE
    target_ref: str = ""
    before_sha: str = ""
    after_sha: str = ""
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
            error=data.get("error", ""),
        )


@dataclass
class JobRecord:
    job_id: str
    issue: int | None = None
    workspace: str = "~/.ptq_workspace"
    legacy_machine: str | None = None
    pr_url: str | None = None
    human_note: str | None = None
    pr_title: str | None = None
    submission_mode: SubmissionMode = SubmissionMode.PULL_REQUEST
    stack_base: str = "main"
    stack_pr_urls: list[str] = field(default_factory=list)
    rebase: RebaseInfo | None = None
    name: str | None = None
    repo: str = "pytorch"

    @property
    def rebase_info(self) -> RebaseInfo:
        if self.rebase is None:
            self.rebase = RebaseInfo()
        return self.rebase

    def to_dict(self) -> dict:
        d: dict = {"issue": self.issue}
        d["workspace"] = self.workspace
        if self.legacy_machine:
            d["machine"] = self.legacy_machine
        if self.pr_url:
            d["pr_url"] = self.pr_url
        if self.human_note:
            d["human_note"] = self.human_note
        if self.pr_title:
            d["pr_title"] = self.pr_title
        if self.submission_mode != SubmissionMode.PULL_REQUEST:
            d["submission_mode"] = self.submission_mode.value
        if self.stack_base != "main":
            d["stack_base"] = self.stack_base
        if self.stack_pr_urls:
            d["stack_pr_urls"] = self.stack_pr_urls
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
        rebase_data = data.get("rebase")
        return cls(
            job_id=job_id,
            issue=data.get("issue"),
            workspace=data.get("workspace", "~/.ptq_workspace"),
            legacy_machine=data.get("machine"),
            pr_url=data.get("pr_url"),
            human_note=data.get("human_note"),
            pr_title=data.get("pr_title"),
            submission_mode=SubmissionMode(
                data.get("submission_mode", SubmissionMode.PULL_REQUEST.value)
            ),
            stack_base=data.get("stack_base", "main"),
            stack_pr_urls=list(data.get("stack_pr_urls", [])),
            rebase=RebaseInfo.from_dict(rebase_data) if rebase_data else None,
            name=data.get("name"),
            repo=data.get("repo", "pytorch"),
        )


@dataclass
class PRResult:
    url: str
    branch: str
