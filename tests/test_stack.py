from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from ptq.application.stack_service import (
    StackCommit,
    StackStatus,
    ghstack_remote,
    initialize_stack,
    inspect_stack,
    parse_commits,
    submit_stack,
    validate_submit,
)
from ptq.domain.models import JobRecord, PtqError, SubmissionMode
from ptq.infrastructure.job_repository import JobRepository
from ptq.ssh import LocalBackend


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> CompletedProcess[str]:
    return CompletedProcess("", returncode, stdout, stderr)


def make_repo(tmp_path: Path) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(
        JobRecord(
            job_id="stack-job",
            local=True,
            workspace="/workspace",
        )
    )
    return repo


def mark_as_stack(repo: JobRepository, *, base: str = "main") -> None:
    job = repo.get("stack-job")
    job.submission_mode = SubmissionMode.GHSTACK
    job.stack_base = base
    repo.save(job)


def git_log(*messages: tuple[str, str, str]) -> str:
    return "".join(
        f"{sha}\x1f{subject}\x1f{body}\x1e" for sha, subject, body in messages
    )


def make_backend(
    *, dirty: bool = False, detached: bool = False, branch: str = "feature-stack"
) -> MagicMock:
    backend = MagicMock()
    backend.workspace = "/workspace"

    def run(cmd: str, check: bool = True):
        if "rev-parse --verify --quiet origin/main" in cmd:
            return completed("base-sha\n")
        if "symbolic-ref --short --quiet HEAD" in cmd:
            return completed("", 1) if detached else completed(f"{branch}\n")
        if "status --porcelain" in cmd:
            return completed(" M changed.py\n" if dirty else "")
        if "rev-list --merges" in cmd:
            return completed()
        if "log --reverse" in cmd:
            return completed(git_log(("abc123", "First change", "First change\n")))
        if cmd.startswith("command -v ghstack"):
            return completed("/usr/bin/ghstack\n")
        if "STACK_CONTEXT.md" in cmd:
            return completed()
        return completed()

    backend.run.side_effect = run
    return backend


def test_stack_workflow_uses_a_real_git_worktree(tmp_path):
    workspace = tmp_path / "workspace with space"
    worktree = workspace / "jobs" / "stack-job" / "pytorch"
    worktree.mkdir(parents=True)
    (worktree / ".ghstackrc").write_text("[ghstack]\nremote_name = origin\n")
    commands = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", "PTQ Test"],
        ["git", "config", "user.email", "ptq@example.com"],
    ]
    for command in commands:
        subprocess.run(command, cwd=worktree, check=True, capture_output=True)
    (worktree / "file.txt").write_text("base\n")
    subprocess.run(
        ["git", "add", "file.txt", ".ghstackrc"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Base"], cwd=worktree, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "switch", "-c", "feature-stack"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    (worktree / "file.txt").write_text("feature\n")
    subprocess.run(
        ["git", "commit", "-am", "First change"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )

    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(JobRecord(job_id="stack-job", local=True, workspace=str(workspace)))
    status = inspect_stack(repo, "stack-job")

    assert status.branch == "feature-stack"
    assert status.remote == "origin"
    assert status.base_ref == "origin/main"
    assert status.dirty is False
    assert [commit.subject for commit in status.commits] == ["First change"]

    initialize_stack(repo, "stack-job")
    job_dir = workspace / "jobs" / "stack-job"
    assert repo.get("stack-job").submission_mode == SubmissionMode.GHSTACK
    assert "ptq stack submit stack-job" in (job_dir / "STACK_CONTEXT.md").read_text()


def test_inspect_stack_fails_closed_when_git_query_fails(tmp_path):
    repo = make_repo(tmp_path)
    backend = make_backend()
    original = backend.run.side_effect

    def with_failed_status(cmd: str, check: bool = True):
        if "status --porcelain" in cmd:
            return completed(returncode=128, stderr="not a worktree")
        return original(cmd, check)

    backend.run.side_effect = with_failed_status
    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="not a worktree"),
    ):
        inspect_stack(repo, "stack-job")


def test_repo_config_without_remote_does_not_fall_back_to_user_config(
    tmp_path, monkeypatch
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    (worktree / ".ghstackrc").write_text("[ghstack]\ngithub_url = github.com\n")
    user_config = tmp_path / "user-ghstackrc"
    user_config.write_text("[ghstack]\nremote_name = upstream\n")
    monkeypatch.setenv("GHSTACKRC_PATH", str(user_config))

    assert ghstack_remote(LocalBackend(), str(worktree)) == "origin"


def test_inspect_stack_uses_ghstack_configured_remote(tmp_path):
    repo = make_repo(tmp_path)
    backend = make_backend()
    original = backend.run.side_effect

    def with_upstream(cmd: str, check: bool = True):
        if "GHSTACKRC_PATH" in cmd:
            return completed("/workspace/.ghstackrc")
        if "cat /workspace/.ghstackrc" in cmd:
            return completed("[ghstack]\nremote_name = upstream\n")
        if "rev-parse --verify --quiet upstream/main" in cmd:
            return completed("base-sha\n")
        return original(cmd, check)

    backend.run.side_effect = with_upstream
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        status = inspect_stack(repo, "stack-job")

    assert status.remote == "upstream"
    assert status.base_ref == "upstream/main"


def test_parse_commits_accepts_legacy_pull_request_trailers():
    commits = parse_commits(
        git_log(
            (
                "abc123",
                "Legacy stack",
                "ghstack-source-id: source-1\n"
                "Pull Request resolved: https://github.com/pytorch/pytorch/pull/99\n",
            )
        )
    )
    assert commits[0].pr_url == "https://github.com/pytorch/pytorch/pull/99"


def test_initialize_stack_rejects_job_with_conventional_pr(tmp_path):
    repo = make_repo(tmp_path)
    job = repo.get("stack-job")
    job.pr_url = "https://github.com/pytorch/pytorch/pull/99"
    repo.save(job)
    backend = make_backend()

    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="conventional PR"),
    ):
        initialize_stack(repo, "stack-job")

    backend.run.assert_not_called()


def test_initialize_stack_preserves_configured_base(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo, base="release/2.8")
    backend = make_backend()

    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        status = initialize_stack(repo, "stack-job")

    assert status.base == "release/2.8"
    assert repo.get("stack-job").stack_base == "release/2.8"


@pytest.mark.parametrize("starting_point", ["base", "detached"])
def test_initialize_stack_creates_branch_from_base_or_detached_head(
    tmp_path, starting_point
):
    repo = make_repo(tmp_path)
    backend = (
        make_backend(branch="main")
        if starting_point == "base"
        else make_backend(detached=True)
    )
    original = backend.run.side_effect
    attached = False

    def with_switch(cmd: str, check: bool = True):
        nonlocal attached
        if "symbolic-ref --short --quiet HEAD" in cmd and attached:
            return completed("ptq-stack/stack-job\n")
        if "show-ref --verify" in cmd:
            return completed(returncode=1)
        if "git switch -c ptq-stack/stack-job" in cmd:
            attached = True
            return completed()
        return original(cmd, check)

    backend.run.side_effect = with_switch
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        status = initialize_stack(repo, "stack-job")

    assert status.branch == "ptq-stack/stack-job"
    assert repo.get("stack-job").submission_mode == SubmissionMode.GHSTACK


def test_initialize_stack_rejects_stale_existing_stack_branch(tmp_path):
    repo = make_repo(tmp_path)
    backend = make_backend(detached=True)
    original = backend.run.side_effect

    def with_stale_branch(cmd: str, check: bool = True):
        if "rev-parse HEAD" in cmd:
            return completed("current-head\n")
        if "show-ref --verify" in cmd:
            return completed()
        if "rev-parse refs/heads/ptq-stack/stack-job" in cmd:
            return completed("stale-head\n")
        return original(cmd, check)

    backend.run.side_effect = with_stale_branch
    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="different commit"),
    ):
        initialize_stack(repo, "stack-job")


@pytest.mark.parametrize(
    ("commits", "has_merges", "message"),
    [
        ((), False, "no commits"),
        ((("abc", "Change"),), True, "linear history"),
    ],
)
def test_submit_validation_rejects_empty_or_merge_stacks(commits, has_merges, message):
    stack_commits = tuple(
        StackCommit(sha=sha, subject=subject) for sha, subject in commits
    )
    status = StackStatus(
        branch="feature-stack",
        base="main",
        remote="origin",
        base_ref="origin/main",
        dirty=False,
        has_merges=has_merges,
        commits=stack_commits,
    )
    with pytest.raises(PtqError, match=message):
        validate_submit(status)


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (make_backend(dirty=True), "uncommitted changes"),
        (make_backend(detached=True), "detached HEAD"),
    ],
)
def test_submit_rejects_unsafe_worktree_state(tmp_path, backend, message):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)

    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match=message),
    ):
        submit_stack(repo, "stack-job")

    assert all(
        "ghstack submit" not in call.args[0] for call in backend.run.call_args_list
    )


def test_submit_requires_stack_initialization_without_mutating_job(tmp_path):
    repo = make_repo(tmp_path)
    backend = make_backend()

    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="stack init"),
    ):
        submit_stack(repo, "stack-job")

    assert repo.get("stack-job").submission_mode == SubmissionMode.PULL_REQUEST
    backend.run.assert_not_called()


def test_submit_stops_when_base_fetch_fails(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    backend = make_backend()
    original = backend.run.side_effect

    def with_failed_fetch(cmd: str, check: bool = True):
        if "fetch --prune origin" in cmd:
            return completed(returncode=1, stderr="network failure")
        return original(cmd, check)

    backend.run.side_effect = with_failed_fetch
    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="network failure"),
    ):
        submit_stack(repo, "stack-job")

    assert all(
        "ghstack submit" not in call.args[0] for call in backend.run.call_args_list
    )


def test_submit_runs_ghstack_and_persists_top_pr(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    backend = MagicMock()
    backend.workspace = "/workspace"
    submitted = False

    def run(cmd: str, check: bool = True):
        nonlocal submitted
        if "GHSTACKRC_PATH" in cmd or cmd.startswith("cat "):
            return completed()
        if "rev-parse --verify --quiet origin/main" in cmd:
            return completed("base-sha\n")
        if "symbolic-ref --short --quiet HEAD" in cmd:
            return completed("feature-stack\n")
        if (
            "status --porcelain" in cmd
            or "rev-list --merges" in cmd
            or "fetch --prune origin" in cmd
        ):
            return completed()
        if "log --reverse" in cmd:
            if not submitted:
                return completed(
                    git_log(
                        ("old1", "First change", "First change\n"),
                        ("old2", "Second change", "Second change\n"),
                    )
                )
            return completed(
                git_log(
                    (
                        "new1",
                        "First change",
                        "First change\n\nghstack-source-id: source-1\n"
                        "Pull-Request: https://github.com/pytorch/pytorch/pull/101\n",
                    ),
                    (
                        "new2",
                        "Second change",
                        "Second change\n\nghstack-source-id: source-2\n"
                        "Pull-Request: https://github.com/pytorch/pytorch/pull/102\n",
                    ),
                )
            )
        if cmd.startswith("command -v ghstack"):
            return completed("/usr/bin/ghstack\n")
        if "ghstack submit" in cmd:
            submitted = True
            return completed("Submitted stack\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    backend.run.side_effect = run
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        result = submit_stack(repo, "stack-job", draft=True)

    submit_call = next(
        call.args[0]
        for call in backend.run.call_args_list
        if "ghstack submit" in call.args[0]
    )
    assert "ghstack submit --stack -B main --draft HEAD" in submit_call
    assert "--update-fields" not in submit_call
    commands = [call.args[0] for call in backend.run.call_args_list]
    fetch_call = next(cmd for cmd in commands if "fetch --prune" in cmd)
    assert commands.index(fetch_call) < commands.index(submit_call)
    assert [commit.sha for commit in result.status.commits] == ["new1", "new2"]

    saved = repo.get("stack-job")
    assert saved.pr_url == "https://github.com/pytorch/pytorch/pull/102"
    assert saved.pr_title == "Second change"
    assert saved.submission_mode == SubmissionMode.GHSTACK


def test_submit_preserves_cached_title_without_metadata_update(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    job = repo.get("stack-job")
    job.pr_url = "https://github.com/pytorch/pytorch/pull/101"
    job.pr_title = "GitHub-edited title"
    repo.save(job)
    backend = make_backend()
    original = backend.run.side_effect

    def with_submit(cmd: str, check: bool = True):
        if "ghstack submit" in cmd:
            return completed()
        if "log --reverse" in cmd:
            return completed(
                git_log(
                    (
                        "new1",
                        "Local title",
                        "ghstack-source-id: source-1\n"
                        "Pull-Request: https://github.com/pytorch/pytorch/pull/101\n",
                    )
                )
            )
        return original(cmd, check)

    backend.run.side_effect = with_submit
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        submit_stack(repo, "stack-job")

    assert repo.get("stack-job").pr_title == "GitHub-edited title"


def test_metadata_update_is_explicit(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    backend = make_backend()
    original = backend.run.side_effect

    def with_submit(cmd: str, check: bool = True):
        if "ghstack submit" in cmd:
            return completed()
        if "log --reverse" in cmd:
            return completed(
                git_log(
                    (
                        "new1",
                        "First change",
                        "ghstack-source-id: source-1\n"
                        "Pull-Request: https://github.com/pytorch/pytorch/pull/101\n",
                    )
                )
            )
        return original(cmd, check)

    backend.run.side_effect = with_submit
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        submit_stack(repo, "stack-job", update_metadata=True)

    submit_call = next(
        call.args[0]
        for call in backend.run.call_args_list
        if "ghstack submit" in call.args[0]
    )
    assert "--update-fields" in submit_call


def test_submit_reports_missing_ghstack(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    backend = make_backend()
    original = backend.run.side_effect

    def without_ghstack(cmd: str, check: bool = True):
        if cmd.startswith("command -v ghstack"):
            return completed(returncode=1)
        return original(cmd, check)

    backend.run.side_effect = without_ghstack
    with (
        patch("ptq.application.stack_service.backend_for_job", return_value=backend),
        pytest.raises(PtqError, match="not installed"),
    ):
        submit_stack(repo, "stack-job")

    saved = repo.get("stack-job")
    assert saved.submission_mode == SubmissionMode.GHSTACK
    assert saved.stack_base == "main"
    assert all(
        "STACK_CONTEXT.md" not in call.args[0] for call in backend.run.call_args_list
    )
