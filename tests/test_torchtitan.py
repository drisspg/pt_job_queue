"""Tests for torchtitan repo support: domain models, issue extraction, prompts."""

from __future__ import annotations

from ptq.agent import build_adhoc_prompt, build_system_prompt
from ptq.domain.models import JobRecord, RunRequest
from ptq.issue import extract_repro_script


# -- Domain models --


class TestJobRecordRepo:
    def test_default_repo_is_pytorch(self):
        record = JobRecord(job_id="j1")
        assert record.repo == "pytorch"

    def test_repo_roundtrip(self):
        record = JobRecord(job_id="j1", repo="torchtitan")
        d = record.to_dict()
        assert d["repo"] == "torchtitan"
        restored = JobRecord.from_dict("j1", d)
        assert restored.repo == "torchtitan"

    def test_default_repo_omitted_from_dict(self):
        record = JobRecord(job_id="j1", repo="pytorch")
        d = record.to_dict()
        assert "repo" not in d

    def test_from_dict_missing_repo_defaults_to_pytorch(self):
        record = JobRecord.from_dict("j1", {})
        assert record.repo == "pytorch"


class TestRunRequestRepo:
    def test_default_repo_is_pytorch(self):
        req = RunRequest(message="hello")
        assert req.repo == "pytorch"

    def test_torchtitan_repo(self):
        req = RunRequest(message="hello", repo="torchtitan")
        assert req.repo == "torchtitan"


# -- Issue extraction --


class TestExtractReproTorchtitan:
    def test_finds_torchtitan_import(self):
        issue = {
            "body": "```python\nimport torchtitan\ncrash()\n```",
        }
        result = extract_repro_script(issue, import_hint="import torchtitan")
        assert result == "import torchtitan\ncrash()"

    def test_torch_import_fallback(self):
        """Even with torchtitan hint, blocks with 'import torch' are still found."""
        issue = {
            "body": "```python\nimport torch\ntorch.distributed.init_process_group()\n```",
        }
        result = extract_repro_script(issue, import_hint="import torchtitan")
        assert "import torch" in result

    def test_skips_unrelated_blocks(self):
        issue = {
            "body": "```python\nimport numpy\nprint('hi')\n```",
        }
        result = extract_repro_script(issue, import_hint="import torchtitan")
        assert result is None


# -- Agent prompts --


class TestTorchtitanPrompts:
    def test_system_prompt_uses_torchtitan_template(self):
        issue_data = {
            "title": "FSDP crash",
            "body": "OOM during training",
            "labels": [],
            "comments": [],
        }
        result = build_system_prompt(issue_data, 2818, "j-2818", "/ws", repo="torchtitan")
        assert "torchtitan" in result
        assert "j-2818" in result
        assert "spin fixlint" not in result
        assert "create_worktree.py" not in result

    def test_adhoc_prompt_uses_torchtitan_template(self):
        result = build_adhoc_prompt("fix FSDP", "j-123", "/ws", repo="torchtitan")
        assert "torchtitan" in result
        assert "j-123" in result
        assert "spin fixlint" not in result

    def test_pytorch_prompts_unchanged(self):
        issue_data = {
            "title": "Bug",
            "body": "desc",
            "labels": [],
            "comments": [],
        }
        result = build_system_prompt(issue_data, 42, "j-42", "/ws", repo="pytorch")
        assert "spin fixlint" in result

        result = build_adhoc_prompt("fix oom", "j-42", "/ws", repo="pytorch")
        assert "spin fixlint" in result
