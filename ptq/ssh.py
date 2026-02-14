from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RemoteBackend:
    machine: str
    workspace: str = "~/ptq_workspace"
    ssh_opts: list[str] = field(
        default_factory=lambda: ["-o", "StrictHostKeyChecking=no"]
    )

    @staticmethod
    def _with_path(cmd: str) -> str:
        return f'source ~/.profile 2>/dev/null; source ~/.bashrc 2>/dev/null; source ~/.zshrc 2>/dev/null; export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" && {cmd}'

    def run(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", *self.ssh_opts, self.machine, self._with_path(cmd)],
            capture_output=True,
            text=True,
            check=check,
        )

    def run_streaming(
        self, cmd: str, follow: bool = True
    ) -> subprocess.Popen[str] | subprocess.CompletedProcess[str]:
        if follow:
            return subprocess.Popen(
                ["ssh", "-tt", *self.ssh_opts, self.machine, self._with_path(cmd)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        nohup_cmd = f"nohup {cmd} &"
        return self.run(nohup_cmd, check=False)

    def copy_to(self, local_path: Path, remote_path: str) -> None:
        subprocess.run(
            ["scp", *self.ssh_opts, str(local_path), f"{self.machine}:{remote_path}"],
            check=True,
            capture_output=True,
        )

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", *self.ssh_opts, f"{self.machine}:{remote_path}", str(local_path)],
            check=True,
            capture_output=True,
        )


@dataclass
class LocalBackend:
    workspace: str = "~/.ptq_workspace"

    @property
    def _workspace_path(self) -> Path:
        return Path(self.workspace).expanduser()

    def run(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["zsh", "-c", cmd],
            capture_output=True,
            text=True,
            check=check,
        )

    def run_streaming(
        self, cmd: str, follow: bool = True
    ) -> subprocess.Popen[str] | subprocess.CompletedProcess[str]:
        if follow:
            return subprocess.Popen(
                ["zsh", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        nohup_cmd = f"nohup {cmd} &"
        return self.run(nohup_cmd, check=False)

    def copy_to(self, local_path: Path, remote_path: str) -> None:
        dest = Path(remote_path.replace("~", str(Path.home())))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        src = Path(remote_path.replace("~", str(Path.home())))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)


Backend = RemoteBackend | LocalBackend


def load_jobs_db() -> dict:
    path = Path.home() / ".ptq" / "jobs.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_jobs_db(db: dict) -> None:
    path = Path.home() / ".ptq" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2))


def backend_for_job(job_id: str) -> Backend:
    db = load_jobs_db()
    entry = db.get(job_id)
    if not entry:
        raise SystemExit(f"Unknown job: {job_id}")
    if entry.get("local"):
        return LocalBackend(workspace=entry.get("workspace", "~/.ptq_workspace"))
    return RemoteBackend(
        machine=entry["machine"], workspace=entry.get("workspace", "~/ptq_workspace")
    )
