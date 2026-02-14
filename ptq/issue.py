from __future__ import annotations

import json
import re
import subprocess


def fetch_issue(issue_number: int, repo: str = "pytorch/pytorch") -> dict:
    result = subprocess.run(
        [
            "gh", "issue", "view", str(issue_number),
            "--repo", repo,
            "--json", "title,body,comments,labels",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def extract_repro_script(issue_data: dict) -> str | None:
    body = issue_data.get("body", "") or ""
    for comment in issue_data.get("comments", []):
        body += "\n" + (comment.get("body", "") or "")

    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
    for block in code_blocks:
        if "import torch" in block:
            return block.strip()
    return None


def format_issue_context(issue_data: dict, issue_number: int) -> str:
    title = issue_data.get("title", "")
    body = issue_data.get("body", "") or ""
    labels = [l.get("name", "") for l in issue_data.get("labels", [])]

    lines = [
        f"# Issue #{issue_number}: {title}",
        "",
        f"**Labels**: {', '.join(labels) if labels else 'none'}",
        "",
        "## Description",
        "",
        body,
    ]

    comments = issue_data.get("comments", [])
    if comments:
        lines.extend(["", "## Comments", ""])
        for i, comment in enumerate(comments, 1):
            author = comment.get("author", {}).get("login", "unknown")
            comment_body = comment.get("body", "")
            lines.extend([f"### Comment {i} by @{author}", "", comment_body, ""])

    return "\n".join(lines)
