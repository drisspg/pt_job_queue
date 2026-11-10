from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from ptq.application.stack_service import (
    StackCommit,
    StackStatus,
    choose_stack_tool,
    ghstack_remote,
    initialize_stack,
    inspect_stack,
    parse_commits,
    persist_submitted_stack,
    submission_tool,
    submit_stack,
    validate_submit,
)
from ptq.domain.models import JobRecord, PtqError, StackTool, SubmissionMode
from ptq.infrastructure.backends import LocalBackend
from ptq.infrastructure.job_repository import JobRepository


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> CompletedProcess[str]:
    return CompletedProcess("", returncode, stdout, stderr)


def make_repo(tmp_path: Path) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(
        JobRecord(
            job_id="stack-job",
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
    repo.save(JobRecord(job_id="stack-job", workspace=str(workspace)))
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


def git(worktree: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=worktree, check=True, capture_output=True)


def make_conventional_pr_job(tmp_path: Path) -> tuple[JobRepository, Path, Path]:
    """A real worktree whose branch ptq/42 is pushed as a conventional PR."""
    workspace = tmp_path / "workspace"
    worktree = workspace / "jobs" / "stack-job" / "pytorch"
    worktree.mkdir(parents=True)
    (worktree / ".ghstackrc").write_text("[ghstack]\nremote_name = origin\n")
    git(worktree, "init", "-b", "main")
    git(worktree, "config", "user.name", "PTQ Test")
    git(worktree, "config", "user.email", "ptq@example.com")
    (worktree / "file.txt").write_text("base\n")
    git(worktree, "add", "file.txt", ".ghstackrc")
    git(worktree, "commit", "-m", "Base")
    git(worktree, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(worktree, "switch", "-c", "ptq/42")
    (worktree / "file.txt").write_text("pr\n")
    git(worktree, "commit", "-am", "Conventional PR change")
    git(worktree, "update-ref", "refs/remotes/origin/ptq/42", "HEAD")

    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(
        JobRecord(
            job_id="stack-job",
            issue=42,
            workspace=str(workspace),
            pr_url="https://github.com/pytorch/pytorch/pull/99",
        )
    )
    return repo, workspace, worktree


def test_initialize_stack_builds_on_existing_conventional_pr(tmp_path):
    repo, workspace, worktree = make_conventional_pr_job(tmp_path)

    status = initialize_stack(repo, "stack-job")

    assert status.branch == "ptq-stack/stack-job"
    assert status.base_ref == "origin/ptq/42"
    job = repo.get("stack-job")
    assert job.submission_mode == SubmissionMode.GHSTACK
    assert job.stack_base == "ptq/42"
    assert job.stack_base_pr_url == "https://github.com/pytorch/pytorch/pull/99"
    context = (workspace / "jobs" / "stack-job" / "STACK_CONTEXT.md").read_text()
    assert "https://github.com/pytorch/pytorch/pull/99" in context
    assert "ghstack-mergeability-check" in context

    (worktree / "other.txt").write_text("stacked\n")
    git(worktree, "add", "other.txt")
    git(worktree, "commit", "-m", "Stacked change")
    assert [c.subject for c in inspect_stack(repo, "stack-job").commits] == [
        "Stacked change"
    ]


def test_initialize_stack_rejects_commits_missing_from_conventional_pr(tmp_path):
    repo, _, worktree = make_conventional_pr_job(tmp_path)
    (worktree / "file.txt").write_text("unpushed\n")
    git(worktree, "commit", "-am", "Unpushed change")

    with pytest.raises(PtqError, match="not on origin/ptq/42"):
        initialize_stack(repo, "stack-job")

    assert repo.get("stack-job").submission_mode == SubmissionMode.PULL_REQUEST


def test_submitted_stack_tracks_its_conventional_base_pr(tmp_path):
    repo = make_repo(tmp_path)
    job = repo.get("stack-job")
    job.submission_mode = SubmissionMode.GHSTACK
    job.pr_url = "https://github.com/pytorch/pytorch/pull/99"
    job.stack_base_pr_url = "https://github.com/pytorch/pytorch/pull/99"
    repo.save(job)

    persist_submitted_stack(
        repo,
        "stack-job",
        (
            StackCommit(
                sha="new1",
                subject="Stacked change",
                source_id="source-1",
                pr_url="https://github.com/pytorch/pytorch/pull/101",
            ),
        ),
    )

    saved = repo.get("stack-job")
    assert saved.stack_pr_urls == [
        "https://github.com/pytorch/pytorch/pull/99",
        "https://github.com/pytorch/pytorch/pull/101",
    ]
    assert saved.pr_url == "https://github.com/pytorch/pytorch/pull/101"


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
    logs: list[str] = []

    def run(cmd: str, check: bool = True, stream: bool = False):
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
        result = submit_stack(
            repo,
            "stack-job",
            draft=True,
            stream_output=True,
            log=logs.append,
        )

    submit_call = next(
        call.args[0]
        for call in backend.run.call_args_list
        if "ghstack submit" in call.args[0]
    )
    assert "ghstack submit --stack -B main --draft HEAD" in submit_call
    assert "--update-fields" not in submit_call
    submit_invocation = next(
        call for call in backend.run.call_args_list if "ghstack submit" in call.args[0]
    )
    assert submit_invocation.kwargs["stream"] is True
    assert logs == [
        "Resolving stack configuration...",
        "Fetching origin...",
        "Submitting 2 commit(s) from feature-stack onto origin/main with ghstack...",
        "Refreshing submitted stack metadata...",
    ]
    commands = [call.args[0] for call in backend.run.call_args_list]
    fetch_call = next(cmd for cmd in commands if "fetch --prune" in cmd)
    assert commands.index(fetch_call) < commands.index(submit_call)
    assert [commit.sha for commit in result.status.commits] == ["new1", "new2"]

    saved = repo.get("stack-job")
    assert saved.stack_pr_urls == [
        "https://github.com/pytorch/pytorch/pull/101",
        "https://github.com/pytorch/pytorch/pull/102",
    ]
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


@pytest.mark.parametrize(
    ("env", "sp_path", "expected"),
    [
        ("", "/usr/bin/sp\n", StackTool.SP),
        ("", "", StackTool.GHSTACK),
        ("ghstack", "/usr/bin/sp\n", StackTool.GHSTACK),
        ("sp", "", StackTool.SP),
    ],
)
def test_choose_stack_tool_prefers_installed_sp(monkeypatch, env, sp_path, expected):
    monkeypatch.setenv("PTQ_STACK_TOOL", env)
    backend = MagicMock()
    backend.run.return_value = completed(sp_path, 0 if sp_path else 1)

    assert choose_stack_tool(backend) == expected


def test_choose_stack_tool_rejects_unknown_override(monkeypatch):
    monkeypatch.setenv("PTQ_STACK_TOOL", "stack-pr")
    with pytest.raises(PtqError, match="PTQ_STACK_TOOL"):
        choose_stack_tool(MagicMock())


def test_submission_tool_follows_existing_trailers():
    sp_job = JobRecord(job_id="j", stack_tool=StackTool.SP)
    ghstack_commit = StackCommit(sha="a", subject="A", source_id="s", pr_url="u")
    sp_commit = StackCommit(sha="b", subject="B", pr_url="u", stack_info=True)

    assert submission_tool(sp_job, (ghstack_commit,)) == StackTool.GHSTACK
    assert submission_tool(JobRecord(job_id="j"), (sp_commit,)) == StackTool.SP
    assert submission_tool(sp_job, (StackCommit(sha="c", subject="C"),)) == StackTool.SP
    assert submission_tool(JobRecord(job_id="j"), ()) == StackTool.GHSTACK
    with pytest.raises(PtqError, match="mixes"):
        submission_tool(sp_job, (ghstack_commit, sp_commit))


def test_submit_with_sp_publishes_a_native_stack(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    job = repo.get("stack-job")
    job.stack_tool = StackTool.SP
    repo.save(job)
    submitted = False

    def run(cmd: str, check: bool = True, stream: bool = False):
        nonlocal submitted
        if "rev-parse --verify --quiet origin/main" in cmd:
            return completed("base-sha\n")
        if "symbolic-ref --short --quiet HEAD" in cmd:
            return completed("feature-stack\n")
        if cmd == "command -v sp":
            return completed("/usr/bin/sp\n")
        if "log --reverse" in cmd:
            body = (
                "Change\n\nstack-info: PR: https://github.com/pytorch/pytorch/pull/7, "
                "branch: me/stack/1\n"
                if submitted
                else "Change\n"
            )
            return completed(git_log(("abc", "Change", body)))
        if "sp submit" in cmd:
            submitted = True
            return completed("ok\n")
        return completed()

    backend = MagicMock(workspace="/workspace")
    backend.run.side_effect = run
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        submit_stack(repo, "stack-job", draft=True)

    commands = [call.args[0] for call in backend.run.call_args_list]
    assert any("/usr/bin/sp submit -R origin -T main -d" in cmd for cmd in commands)
    assert not any("ghstack submit" in cmd for cmd in commands)
    saved = repo.get("stack-job")
    assert saved.stack_pr_urls == ["https://github.com/pytorch/pytorch/pull/7"]
    assert saved.pr_url == "https://github.com/pytorch/pytorch/pull/7"


def make_pushed_conventional_pr(tmp_path: Path) -> tuple[JobRepository, Path, Path]:
    """A worktree whose two-commit branch ptq/42 is a conventional PR on a real remote."""
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        check=True,
        capture_output=True,
    )
    repo, workspace, worktree = make_conventional_pr_job(tmp_path)
    (worktree / "file.txt").write_text("pr follow-up\n")
    git(worktree, "commit", "-am", "PR follow-up")
    git(worktree, "remote", "add", "origin", str(remote))
    git(worktree, "push", "origin", "main", "ptq/42")
    git(worktree, "fetch", "origin")
    return repo, worktree, remote


@pytest.fixture
def open_pr(monkeypatch):
    monkeypatch.setenv("PTQ_STACK_TOOL", "sp")
    state = {"value": "open"}
    monkeypatch.setattr(
        "ptq.application.stack_service.get_pr_state",
        lambda *args, **kwargs: state["value"],
    )
    return state


def remote_ref(remote: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_sp_stack_init_adopts_conventional_pr_as_bottom_layer(tmp_path, open_pr):
    repo, worktree, remote = make_pushed_conventional_pr(tmp_path)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    status = initialize_stack(repo, "stack-job")

    assert status.branch == "ptq-stack/stack-job"
    assert status.base_ref == "origin/main"
    assert [c.pr_url for c in status.commits] == [
        "https://github.com/pytorch/pytorch/pull/99"
    ]
    assert status.commits[0].stack_info
    remote_head = subprocess.run(
        ["git", "rev-parse", "ptq/42", "ptq/42^{tree}"],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert remote_head == [status.commits[0].sha, tree.strip()]
    job = repo.get("stack-job")
    assert job.stack_tool == StackTool.SP
    assert job.stack_base == "main"
    assert job.stack_base_pr_url is None
    context = (
        tmp_path / "workspace" / "jobs" / "stack-job" / "STACK_CONTEXT.md"
    ).read_text()
    assert (
        "The bottom commit is the existing PR https://github.com/pytorch/pytorch/pull/99"
        in context
    )


def test_sp_stack_init_resumes_after_an_interrupted_adoption(tmp_path, open_pr):
    repo, _, remote = make_pushed_conventional_pr(tmp_path)
    first = initialize_stack(repo, "stack-job")
    job = repo.get("stack-job")
    job.submission_mode = SubmissionMode.PULL_REQUEST
    job.stack_tool = None
    repo.save(job)

    again = initialize_stack(repo, "stack-job")

    assert [c.sha for c in again.commits] == [c.sha for c in first.commits]
    assert remote_ref(remote, "ptq/42") == first.commits[0].sha
    assert repo.get("stack-job").stack_tool == StackTool.SP


def test_sp_stack_init_leaves_a_closed_pr_untouched(tmp_path, open_pr):
    open_pr["value"] = "closed"
    repo, worktree, remote = make_pushed_conventional_pr(tmp_path)
    before = remote_ref(remote, "ptq/42")

    with pytest.raises(PtqError, match="not open"):
        initialize_stack(repo, "stack-job")

    assert remote_ref(remote, "ptq/42") == before


def test_sp_stack_init_refuses_unpushed_conventional_pr(tmp_path, open_pr):
    repo, worktree, remote = make_pushed_conventional_pr(tmp_path)
    (worktree / "file.txt").write_text("unpushed\n")
    git(worktree, "commit", "-am", "Unpushed")

    with pytest.raises(PtqError, match="pushed head"):
        initialize_stack(repo, "stack-job")

    assert repo.get("stack-job").submission_mode == SubmissionMode.PULL_REQUEST


def test_submit_trusts_ghstack_trailers_when_sp_delegates(tmp_path):
    repo = make_repo(tmp_path)
    mark_as_stack(repo)
    job = repo.get("stack-job")
    job.stack_tool = StackTool.SP
    repo.save(job)
    submitted = False

    def run(cmd: str, check: bool = True, stream: bool = False):
        nonlocal submitted
        if "rev-parse --verify --quiet origin/main" in cmd:
            return completed("base-sha\n")
        if "symbolic-ref --short --quiet HEAD" in cmd:
            return completed("feature-stack\n")
        if cmd == "command -v sp":
            return completed("/usr/bin/sp\n")
        if "log --reverse" in cmd:
            body = (
                "Change\n\nghstack-source-id: s1\n"
                "Pull-Request: https://github.com/pytorch/pytorch/pull/8\n"
                if submitted
                else "Change\n"
            )
            return completed(git_log(("abc", "Change", body)))
        if "sp submit" in cmd:
            submitted = True
            return completed()
        return completed()

    backend = MagicMock(workspace="/workspace")
    backend.run.side_effect = run
    with patch("ptq.application.stack_service.backend_for_job", return_value=backend):
        submit_stack(repo, "stack-job")

    saved = repo.get("stack-job")
    assert saved.stack_pr_urls == ["https://github.com/pytorch/pytorch/pull/8"]
    assert saved.stack_tool == StackTool.GHSTACK
