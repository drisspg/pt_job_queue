from __future__ import annotations

import re
from pathlib import Path

PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "investigate.md"
).read_text()

ADHOC_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "adhoc.md"
).read_text()

RESERVED_HEADER_RE = re.compile(r"x-anthropic-\S+", re.IGNORECASE)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\r")

MAX_OUTPUT_LINES = 30

DEFAULT_MESSAGE = (
    "Investigate and fix the PyTorch issue described in your system prompt."
)


def _sanitize_for_api(text: str) -> str:
    return RESERVED_HEADER_RE.sub("[redacted-header]", text)


def build_system_prompt(
    issue_data: dict, issue_number: int, job_id: str, workspace: str
) -> str:
    from ptq.issue import format_issue_context

    return _sanitize_for_api(
        PROMPT_TEMPLATE.format(
            job_id=job_id,
            issue_number=issue_number,
            issue_context=format_issue_context(issue_data, issue_number),
            workspace=workspace,
        )
    )


def build_adhoc_prompt(message: str, job_id: str, workspace: str) -> str:
    return _sanitize_for_api(
        ADHOC_PROMPT_TEMPLATE.format(
            job_id=job_id,
            task_description=message,
            workspace=workspace,
        )
    )


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def _truncate(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
