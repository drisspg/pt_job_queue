from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ptq.pr import _build_pr_body, create_pr


class TestBuildPRBody:
    def test_with_report_and_issue(self):
        body = _build_pr_body("The fix", "", 12345)
        assert "The fix" in body
        assert "Fixes #12345" in body

    def test_with_worklog_collapsed(self):
        body = _build_pr_body("Report", "## Run 1\nDid stuff", 100)
        assert "<details>" in body
        assert "<summary>Worklog</summary>" in body
        assert "## Run 1" in body

    def test_adhoc_no_issue(self):
        body = _build_pr_body("Report", "", None)
        assert "Fixes" not in body
        assert "Report" in body

    def test_empty_fallback(self):
        body = _build_pr_body("", "", None)
        assert "ptq" in body.lower()


class TestCreatePR:
    @pytest.fixture()
    def mock_backend(self):
        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/pytorch/pytorch/pull/99999\n",
            stderr="",
        )
        return backend

    def _patches(self, db, mock_backend):
        return (
            patch("ptq.ssh.load_jobs_db", return_value=db),
            patch("ptq.ssh.save_jobs_db"),
            patch("ptq.job.load_jobs_db", return_value=db),
            patch("ptq.job.save_jobs_db"),
            patch("ptq.pr.backend_for_job", return_value=mock_backend),
        )

    def test_creates_pr(self, mock_backend):
        db = {
            "j1": {
                "issue": 176093,
                "runs": 6,
                "local": True,
                "workspace": "~/.ptq_workspace",
                "agent": "claude",
            }
        }
        p1, p2, p3, p4, p5 = self._patches(db, mock_backend)
        with p1, p2, p3, p4, p5:
            result = create_pr("j1")

        assert result.url == "https://github.com/pytorch/pytorch/pull/99999"
        assert result.branch == "ptq/176093"

        run_cmds = [c.args[0] for c in mock_backend.run.call_args_list]
        assert any("git add -A" in c for c in run_cmds)
        assert any("git commit" in c for c in run_cmds)
        assert any("git push" in c for c in run_cmds)
        assert any("gh pr create" in c for c in run_cmds)

    def test_adhoc_branch_name(self, mock_backend):
        db = {
            "j1": {
                "issue": None,
                "runs": 1,
                "local": True,
                "workspace": "~/.ptq_workspace",
                "agent": "claude",
            }
        }
        p1, p2, p3, p4, p5 = self._patches(db, mock_backend)
        with p1, p2, p3, p4, p5:
            result = create_pr("j1")

        assert result.branch == "ptq/j1"

    def test_draft_flag(self, mock_backend):
        db = {
            "j1": {
                "issue": 176093,
                "runs": 1,
                "local": True,
                "workspace": "~/.ptq_workspace",
                "agent": "claude",
            }
        }
        p1, p2, p3, p4, p5 = self._patches(db, mock_backend)
        with p1, p2, p3, p4, p5:
            create_pr("j1", draft=True)

        run_cmds = [c.args[0] for c in mock_backend.run.call_args_list]
        gh_cmd = next(c for c in run_cmds if "gh pr create" in c)
        assert "--draft" in gh_cmd

    def test_custom_title(self, mock_backend):
        db = {
            "j1": {
                "issue": 176093,
                "runs": 1,
                "local": True,
                "workspace": "~/.ptq_workspace",
                "agent": "claude",
            }
        }
        p1, p2, p3, p4, p5 = self._patches(db, mock_backend)
        with p1, p2, p3, p4, p5:
            create_pr("j1", title="Custom title")

        run_cmds = [c.args[0] for c in mock_backend.run.call_args_list]
        commit_cmd = next(c for c in run_cmds if "git commit" in c)
        assert "Custom title" in commit_cmd
