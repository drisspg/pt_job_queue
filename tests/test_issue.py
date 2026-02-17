from __future__ import annotations

import json
from unittest.mock import patch

from ptq.issue import extract_repro_script, fetch_issue, format_issue_context


class TestExtractReproScript:
    def test_python_fenced_block(self):
        issue = {
            "body": "Steps:\n```python\nimport torch\nx = torch.randn(3)\n```\n",
        }
        assert extract_repro_script(issue) == "import torch\nx = torch.randn(3)"

    def test_bare_fenced_block(self):
        issue = {
            "body": "Repro:\n```\nimport torch\nprint(torch.__version__)\n```\n",
        }
        assert extract_repro_script(issue) == "import torch\nprint(torch.__version__)"

    def test_skips_blocks_without_torch(self):
        issue = {
            "body": "```python\nimport numpy\n```\n```python\nimport torch\npass\n```\n",
        }
        assert "import torch" in extract_repro_script(issue)

    def test_finds_in_comments(self):
        issue = {
            "body": "No code here.",
            "comments": [{"body": "```\nimport torch\nfail()\n```"}],
        }
        assert extract_repro_script(issue) == "import torch\nfail()"

    def test_returns_none_no_code_blocks(self):
        assert extract_repro_script({"body": "just text, no code"}) is None

    def test_returns_none_when_body_is_none(self):
        assert extract_repro_script({"body": None}) is None


class TestFormatIssueContext:
    def test_basic_with_labels(self):
        issue = {
            "title": "OOM on H100",
            "body": "Runs out of memory.",
            "labels": [{"name": "bug"}, {"name": "high priority"}],
        }
        result = format_issue_context(issue, 123)
        assert "# Issue #123: OOM on H100" in result
        assert "bug" in result
        assert "high priority" in result
        assert "Runs out of memory." in result

    def test_with_comments(self):
        issue = {
            "title": "Bug",
            "body": "desc",
            "labels": [],
            "comments": [
                {"author": {"login": "alice"}, "body": "I see this too"},
            ],
        }
        result = format_issue_context(issue, 1)
        assert "@alice" in result
        assert "I see this too" in result

    def test_no_labels(self):
        issue = {"title": "T", "body": "B", "labels": []}
        assert "none" in format_issue_context(issue, 1)

    def test_none_body(self):
        issue = {"title": "T", "body": None, "labels": []}
        result = format_issue_context(issue, 1)
        assert "# Issue #1: T" in result


class TestFetchIssue:
    def test_calls_gh_and_parses_json(self):
        fake_data = {"title": "Bug", "body": "desc", "comments": [], "labels": []}
        mock_result = type(
            "R",
            (),
            {
                "stdout": json.dumps(fake_data),
                "returncode": 0,
            },
        )()
        with patch("ptq.issue.subprocess.run", return_value=mock_result) as mock_run:
            result = fetch_issue(12345)
        assert result == fake_data
        args = mock_run.call_args[0][0]
        assert "gh" in args
        assert "12345" in args
        assert "pytorch/pytorch" in args
