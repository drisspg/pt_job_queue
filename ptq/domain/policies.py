from __future__ import annotations

import hashlib
from datetime import datetime


def make_job_id(issue_number: int | None = None, message: str | None = None) -> str:
    date = datetime.now().strftime("%Y%m%d")
    if issue_number is not None:
        return f"{date}-{issue_number}"
    return f"{date}-adhoc-{hashlib.md5((message or 'adhoc').encode()).hexdigest()[:6]}"
