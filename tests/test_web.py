from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ptq.config import AgentModels, Config
from ptq.domain.models import JobRecord
from ptq.infrastructure.job_repository import JobRepository
from ptq.web.app import create_app

TEST_CONFIG = Config(
    default_agent="claude",
    default_model="opus",
    default_max_turns=100,
    machines=["gpu-dev"],
    agent_models={
        "claude": AgentModels(available=[], default="opus"),
        "codex": AgentModels(available=[], default="o3"),
    },
)

SAMPLE_RECORDS = [
    JobRecord(
        job_id="20260217-100001",
        issue=100001,
        runs=2,
        machine="gpu-dev",
        workspace="~/ptq_workspace",
        agent="claude",
        pid=12345,
    ),
    JobRecord(
        job_id="20260218-adhoc-abc123",
        local=True,
        workspace="~/.ptq_workspace",
        agent="codex",
    ),
]


def _make_repo(tmp_path: Path) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in SAMPLE_RECORDS:
        repo.save(r)
    return repo


@pytest.fixture()
def mock_backend():
    backend = MagicMock()
    backend.workspace = "~/ptq_workspace"
    backend.is_pid_alive.return_value = False
    return backend


@pytest.fixture()
def client(tmp_path, mock_backend):
    repo = _make_repo(tmp_path)
    with (
        patch("ptq.web.routes._repo", return_value=repo),
        patch("ptq.web.deps.JobRepository", return_value=repo),
        patch("ptq.web.deps.backend_for_job", return_value=mock_backend),
        patch("ptq.web.routes.backend_for_job", return_value=mock_backend),
        patch("ptq.web.routes.load_config", return_value=TEST_CONFIG),
    ):
        yield TestClient(create_app())


class TestDashboard:
    def test_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text

    def test_shows_job_count(self, client):
        resp = client.get("/dashboard")
        assert "Total Jobs" in resp.text

    def test_shows_machine(self, client):
        resp = client.get("/")
        assert "gpu-dev" in resp.text


class TestJobList:
    def test_renders_all(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert "20260217-100001" in resp.text
        assert "20260218-adhoc-abc123" in resp.text

    def test_filter_running(self, client, mock_backend):
        mock_backend.is_pid_alive.return_value = True
        resp = client.get("/jobs?status_filter=running")
        assert resp.status_code == 200

    def test_filter_stopped(self, client):
        resp = client.get("/jobs?status_filter=stopped")
        assert resp.status_code == 200


class TestJobDetail:
    def test_renders(self, client):
        resp = client.get("/jobs/20260217-100001")
        assert resp.status_code == 200
        assert "20260217-100001" in resp.text
        assert "claude" in resp.text

    def test_unknown_job_404(self, client):
        resp = client.get("/jobs/nonexistent-job")
        assert resp.status_code == 404


class TestNewJobForm:
    def test_renders(self, client):
        resp = client.get("/jobs/new")
        assert resp.status_code == 200
        assert "Launch New Job" in resp.text
        assert "claude" in resp.text
        assert "codex" in resp.text
        assert "gpu-dev" in resp.text
        assert "opus" in resp.text

    def test_agent_models_api_returns_fallback_for_claude(self, client):
        resp = client.get("/api/models/claude")
        assert resp.status_code == 200
        assert "<select" in resp.text
        assert "opus" in resp.text
        assert "sonnet" in resp.text
        assert "haiku" in resp.text

    def test_agent_models_api_returns_text_input_when_no_available(self, client):
        resp = client.get("/api/models/unknown_agent")
        assert resp.status_code == 200
        assert '<input type="text"' in resp.text

    def test_agent_models_api_returns_select_when_available(
        self, tmp_path, mock_backend
    ):
        cfg_with_list = Config(
            default_agent="claude",
            default_model="opus",
            default_max_turns=100,
            machines=[],
            agent_models={
                "claude": AgentModels(available=["opus", "sonnet"], default="opus"),
            },
        )
        repo = _make_repo(tmp_path)
        with (
            patch("ptq.web.routes._repo", return_value=repo),
            patch("ptq.web.deps.JobRepository", return_value=repo),
            patch("ptq.web.deps.backend_for_job", return_value=mock_backend),
            patch("ptq.web.routes.backend_for_job", return_value=mock_backend),
            patch("ptq.web.routes.load_config", return_value=cfg_with_list),
        ):
            client = TestClient(create_app())
            resp = client.get("/api/models/claude")
            assert "<select" in resp.text
            assert "opus" in resp.text
            assert "sonnet" in resp.text

    def test_agent_models_api_uses_cache(self, client):
        with patch("ptq.web.routes.cached_models", return_value=["o3", "o4-mini"]):
            resp = client.get("/api/models/codex")
        assert resp.status_code == 200
        assert "<select" in resp.text
        assert "o3" in resp.text
        assert "o4-mini" in resp.text

    def test_agent_models_api_different_per_agent(self, client):
        def fake_cache(agent):
            return {"cursor": ["opus-4.6", "sonnet-4.6"], "codex": ["o3"]}.get(
                agent, []
            )

        with patch("ptq.web.routes.cached_models", side_effect=fake_cache):
            cursor_resp = client.get("/api/models/cursor")
            codex_resp = client.get("/api/models/codex")
        assert "opus-4.6" in cursor_resp.text
        assert "opus-4.6" not in codex_resp.text
        assert "o3" in codex_resp.text

    def test_agent_models_api_unknown(self, client):
        resp = client.get("/api/models/unknown")
        assert resp.status_code == 200

    def test_missing_issue_returns_422(self, client):
        resp = client.post(
            "/jobs",
            data={
                "task_type": "issue",
                "issue": "",
                "message": "",
                "target_type": "local",
                "machine": "",
                "agent": "claude",
                "model": "opus",
                "max_turns": "100",
            },
        )
        assert resp.status_code == 422
        assert "Issue number is required" in resp.text

    def test_missing_message_returns_422(self, client):
        resp = client.post(
            "/jobs",
            data={
                "task_type": "adhoc",
                "issue": "",
                "message": "",
                "target_type": "local",
                "machine": "",
                "agent": "claude",
                "model": "opus",
                "max_turns": "100",
            },
        )
        assert resp.status_code == 422
        assert "Message is required" in resp.text

    def test_missing_machine_returns_422(self, client):
        resp = client.post(
            "/jobs",
            data={
                "task_type": "adhoc",
                "issue": "",
                "message": "do stuff",
                "target_type": "machine",
                "machine": "",
                "agent": "claude",
                "model": "opus",
                "max_turns": "100",
            },
        )
        assert resp.status_code == 422
        assert "Machine name is required" in resp.text


class TestJobActions:
    def test_kill_redirects(self, client, mock_backend):
        mock_backend.is_pid_alive.return_value = True
        resp = client.post("/jobs/20260217-100001/kill", follow_redirects=False)
        assert resp.status_code == 303

    def test_delete_returns_empty(self, client, mock_backend):
        with patch(
            "ptq.application.job_service.backend_for_job", return_value=mock_backend
        ):
            resp = client.delete("/jobs/20260217-100001")
        assert resp.status_code == 200

    def test_create_pr_redirects(self, client, mock_backend):
        from ptq.domain.models import PRResult

        with patch(
            "ptq.application.pr_service.create_pr",
            return_value=PRResult(
                url="https://github.com/pytorch/pytorch/pull/99", branch="ptq/100001"
            ),
        ):
            resp = client.post(
                "/jobs/20260217-100001/pr",
                data={},
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert "pr_url=" in resp.headers["location"]

    def test_rerun_redirects(self, client, mock_backend):
        with patch(
            "ptq.application.run_service.launch", return_value="20260217-100001"
        ):
            resp = client.post(
                "/jobs/20260217-100001/rerun",
                data={"message": "try a different approach"},
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert "/jobs/20260217-100001" in resp.headers["location"]

    def test_rerun_without_message(self, client, mock_backend):
        with patch(
            "ptq.application.run_service.launch", return_value="20260217-100001"
        ):
            resp = client.post(
                "/jobs/20260217-100001/rerun",
                data={"message": ""},
                follow_redirects=False,
            )
        assert resp.status_code == 303

    def test_rerun_switches_agent(self, client, mock_backend):
        with patch(
            "ptq.application.run_service.launch", return_value="20260217-100001"
        ) as mock:
            resp = client.post(
                "/jobs/20260217-100001/rerun",
                data={"message": "try codex", "agent_type": "codex", "model": "o3"},
                follow_redirects=False,
            )
        assert resp.status_code == 303
        request = mock.call_args.args[2]
        assert request.agent_type == "codex"
        assert request.model == "o3"

    def test_rerun_uses_default_model_when_empty(self, client, mock_backend):
        with patch(
            "ptq.application.run_service.launch", return_value="20260217-100001"
        ) as mock:
            resp = client.post(
                "/jobs/20260217-100001/rerun",
                data={"message": "", "agent_type": "claude", "model": ""},
                follow_redirects=False,
            )
        assert resp.status_code == 303
        request = mock.call_args.args[2]
        assert request.model == "opus"


class TestPartials:
    def test_status_badge(self, client):
        resp = client.get("/jobs/20260217-100001/status")
        assert resp.status_code == 200
        assert "badge" in resp.text

    def test_report_no_content(self, client, mock_backend):
        mock_backend.run.return_value = MagicMock(returncode=1, stdout="")
        resp = client.get("/jobs/20260217-100001/report")
        assert resp.status_code == 200
        assert "No report yet" in resp.text

    def test_diff_no_content(self, client, mock_backend):
        mock_backend.run.return_value = MagicMock(returncode=1, stdout="")
        resp = client.get("/jobs/20260217-100001/diff")
        assert resp.status_code == 200
        assert "No diff yet" in resp.text

    def test_diff_renders_colors(self, client, mock_backend):
        mock_backend.run.return_value = MagicMock(
            returncode=0,
            stdout="--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n context\n",
        )
        resp = client.get("/jobs/20260217-100001/diff")
        assert resp.status_code == 200
        assert "diff-add" in resp.text
        assert "diff-del" in resp.text
        assert "diff-hunk" in resp.text


class TestSSELogs:
    def test_stream_endpoint_exists(self, client, mock_backend):
        mock_backend.run.return_value = MagicMock(returncode=1, stdout="")
        resp = client.get("/jobs/20260217-100001/logs/stream")
        assert resp.status_code == 200
