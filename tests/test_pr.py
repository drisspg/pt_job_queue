from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from ptq.application import pr_service
from ptq.application.pr_service import _build_pr_body, _ensure_ssh_remote, create_pr
from ptq.domain.models import JobRecord, PtqError
from ptq.infrastructure.job_repository import JobRepository


class TestPrState:
    def test_get_pr_state_caches(self):
        backend = MagicMock()
        backend.run.return_value = CompletedProcess("", 0, "OPEN\t\n", "")
        pr_service._pr_state_cache.clear()

        first = pr_service.get_pr_state(
            backend, "https://github.com/pytorch/pytorch/pull/99"
        )
        second = pr_service.get_pr_state(
            backend, "https://github.com/pytorch/pytorch/pull/99"
        )

        assert first == "open"
        assert second == "open"
        assert backend.run.call_count == 1

    def test_get_pr_state_closed_with_merged_at_is_merged(self):
        backend = MagicMock()
        backend.run.return_value = CompletedProcess(
            "", 0, "CLOSED\t2026-03-04T10:00:00Z\n", ""
        )
        pr_service._pr_state_cache.clear()

        state = pr_service.get_pr_state(
            backend, "https://github.com/pytorch/pytorch/pull/100", force_refresh=True
        )
        assert state == "merged"


class TestBuildPrBody:
    def test_human_note_at_top(self):
        body = _build_pr_body("Report", "", issue_number=42, human_note="Trivial fix")
        lines = body.splitlines()
        assert lines[0] == "## Human Note"
        assert lines[1] == "Trivial fix"
        assert "Fixes #42" in body

    def test_with_report(self):
        body = _build_pr_body("Report here", "", issue_number=42, human_note="Note")
        assert "Report here" in body
        assert "Agent Report" in body

    def test_with_worklog(self):
        body = _build_pr_body(
            "Report", "log entries", issue_number=None, human_note="N"
        )
        assert "<details>" in body
        assert "log entries" in body
        assert "Agent Worklog" in body

    def test_no_agent_content(self):
        body = _build_pr_body("", "", issue_number=None, human_note="Manual fix")
        assert "Manual fix" in body
        assert "Human Note" in body

    def test_issue_reference(self):
        body = _build_pr_body("R", "", issue_number=99, human_note="Fix")
        assert "Fixes #99" in body


class TestEnsureSshRemote:
    def test_converts_https_to_ssh(self):
        backend = MagicMock()
        backend.run.return_value = MagicMock(
            stdout="https://github.com/pytorch/pytorch\n", returncode=0
        )
        _ensure_ssh_remote(backend, "/worktree", lambda _: None)
        set_url_call = [c for c in backend.run.call_args_list if "set-url" in str(c)]
        assert len(set_url_call) == 1
        assert "git@github.com:pytorch/pytorch.git" in str(set_url_call[0])

    def test_already_ssh_no_op(self):
        backend = MagicMock()
        backend.run.return_value = MagicMock(
            stdout="git@github.com:pytorch/pytorch.git\n", returncode=0
        )
        _ensure_ssh_remote(backend, "/worktree", lambda _: None)
        assert all("set-url" not in str(c) for c in backend.run.call_args_list)


class TestCreatePr:
    def _setup(self, tmp_path: Path) -> tuple[JobRepository, MagicMock]:
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="20260217-42",
                issue=42,
                machine="gpu-dev",
                workspace="~/ptq_workspace",
            )
        )
        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"

        def run_side_effect(cmd, check=True):
            if "git remote get-url" in cmd:
                return CompletedProcess("", 0, "git@github.com:pytorch/pytorch.git\n")
            if "git merge-base" in cmd:
                return CompletedProcess("", 0, "abc123\n")
            if "gh pr create" in cmd:
                return CompletedProcess(
                    "", 0, "https://github.com/pytorch/pytorch/pull/99\n"
                )
            return CompletedProcess("", 0, "")

        backend.run = MagicMock(side_effect=run_side_effect)
        return repo, backend

    def test_creates_pr(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            result = create_pr(repo, "20260217-42", human_note="Trivial fix")
        assert result.url == "https://github.com/pytorch/pytorch/pull/99"
        assert result.branch == "ptq/42"
        assert (
            repo.get("20260217-42").pr_url
            == "https://github.com/pytorch/pytorch/pull/99"
        )

    def test_empty_note_raises(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        with (
            patch("ptq.application.pr_service.backend_for_job", return_value=backend),
            pytest.raises(PtqError, match="human note is required"),
        ):
            create_pr(repo, "20260217-42", human_note="")

    def test_custom_title(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            result = create_pr(
                repo, "20260217-42", human_note="Note", title="Custom Title"
            )
        assert result.branch == "ptq/42"

    def test_updates_known_open_pr(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        job = repo.get("20260217-42")
        job.pr_url = "https://github.com/pytorch/pytorch/pull/77"
        repo.save(job)

        def run_side_effect(cmd, check=True):
            if "gh pr view" in cmd:
                return CompletedProcess("", 0, "OPEN\t\n", "")
            if "gh pr edit 'https://github.com/pytorch/pytorch/pull/77'" in cmd:
                return CompletedProcess("", 0, "", "")
            if "git remote get-url" in cmd:
                return CompletedProcess("", 0, "git@github.com:pytorch/pytorch.git\n")
            if "git merge-base" in cmd:
                return CompletedProcess("", 0, "abc123\n")
            if "gh pr create" in cmd:
                return CompletedProcess("", 1, "", "should not create")
            return CompletedProcess("", 0, "")

        backend.run = MagicMock(side_effect=run_side_effect)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            result = create_pr(repo, "20260217-42", human_note="Updated note")
        assert result.url == "https://github.com/pytorch/pytorch/pull/77"
        create_calls = [
            c for c in backend.run.call_args_list if "gh pr create" in str(c)
        ]
        assert not create_calls

    def test_closed_saved_pr_creates_new_pr(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        job = repo.get("20260217-42")
        job.pr_url = "https://github.com/pytorch/pytorch/pull/77"
        repo.save(job)

        def run_side_effect(cmd, check=True):
            if "gh pr view" in cmd:
                return CompletedProcess("", 0, "CLOSED\t\n", "")
            if "git remote get-url" in cmd:
                return CompletedProcess("", 0, "git@github.com:pytorch/pytorch.git\n")
            if "git merge-base" in cmd:
                return CompletedProcess("", 0, "abc123\n")
            if "gh pr create" in cmd:
                return CompletedProcess(
                    "", 0, "https://github.com/pytorch/pytorch/pull/101\n", ""
                )
            return CompletedProcess("", 0, "")

        backend.run = MagicMock(side_effect=run_side_effect)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            result = create_pr(repo, "20260217-42", human_note="Updated note")
        assert result.url == "https://github.com/pytorch/pytorch/pull/101"

    def test_handles_existing_pr(self, tmp_path):
        repo, backend = self._setup(tmp_path)

        def run_side_effect(cmd, check=True):
            if "git remote get-url" in cmd:
                return CompletedProcess("", 0, "git@github.com:pytorch/pytorch.git\n")
            if "git merge-base" in cmd:
                return CompletedProcess("", 0, "abc123\n")
            if "gh pr create" in cmd:
                return CompletedProcess("", 1, "", "already exists")
            if "gh pr list" in cmd:
                return CompletedProcess(
                    "", 0, "https://github.com/pytorch/pytorch/pull/88\n"
                )
            return CompletedProcess("", 0, "")

        backend.run = MagicMock(side_effect=run_side_effect)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            result = create_pr(repo, "20260217-42", human_note="Updated note")
        assert result.url == "https://github.com/pytorch/pytorch/pull/88"
        assert (
            repo.get("20260217-42").pr_url
            == "https://github.com/pytorch/pytorch/pull/88"
        )
        edit_calls = [c for c in backend.run.call_args_list if "gh pr edit" in str(c)]
        assert len(edit_calls) == 1
        assert "Updated note" in str(edit_calls[0])

    def test_failure_raises(self, tmp_path):
        repo, backend = self._setup(tmp_path)

        def run_side_effect(cmd, check=True):
            if "gh pr create" in cmd:
                return CompletedProcess("", 1, "", "auth required")
            if "git merge-base" in cmd:
                return CompletedProcess("", 0, "abc123\n")
            if "git remote get-url" in cmd:
                return CompletedProcess("", 0, "git@github.com:pytorch/pytorch.git\n")
            return CompletedProcess("", 0, "")

        backend.run = MagicMock(side_effect=run_side_effect)
        with (
            patch("ptq.application.pr_service.backend_for_job", return_value=backend),
            pytest.raises(PtqError, match="gh pr create failed"),
        ):
            create_pr(repo, "20260217-42", human_note="Fix")

    def test_pushes_without_force(self, tmp_path):
        repo, backend = self._setup(tmp_path)
        with patch("ptq.application.pr_service.backend_for_job", return_value=backend):
            create_pr(repo, "20260217-42", human_note="Trivial fix")
        push_calls = [
            call
            for call in backend.run.call_args_list
            if "git push -u origin" in str(call)
        ]
        assert len(push_calls) == 1
        assert "--force" not in str(push_calls[0])
