from __future__ import annotations

import shlex
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

    def run(
        self, cmd: str, check: bool = True, stream: bool = False
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict = {"text": True, "check": check}
        if not stream:
            kwargs["capture_output"] = True
        return subprocess.run(
            ["ssh", *self.ssh_opts, self.machine, self._with_path(cmd)],
            **kwargs,
        )

    def launch_background(self, cmd: str, log_file: str) -> int | None:
        bg_cmd = f"nohup zsh -c {shlex.quote(cmd)} > {log_file} 2>&1 & echo $!"
        result = self.run(bg_cmd, check=False)
        pid_str = (
            result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        )
        return int(pid_str) if pid_str.isdigit() else None

    def is_pid_alive(self, pid: int) -> bool:
        result = self.run(f"kill -0 {pid} 2>/dev/null && echo alive", check=False)
        return "alive" in result.stdout

    def kill_pid(self, pid: int) -> bool:
        result = self.run(f"kill {pid} 2>/dev/null", check=False)
        return result.returncode == 0

    def tail_log(self, log_file: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                "ssh",
                *self.ssh_opts,
                self.machine,
                self._with_path(f"tail -f {log_file}"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

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

    def run(
        self, cmd: str, check: bool = True, stream: bool = False
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict = {"text": True, "check": check}
        if not stream:
            kwargs["capture_output"] = True
        return subprocess.run(
            ["zsh", "-c", cmd],
            **kwargs,
        )

    def launch_background(self, cmd: str, log_file: str) -> int | None:
        bg_cmd = f"nohup zsh -c {shlex.quote(cmd)} > {log_file} 2>&1 & echo $!"
        result = self.run(bg_cmd, check=False)
        pid_str = (
            result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        )
        return int(pid_str) if pid_str.isdigit() else None

    def is_pid_alive(self, pid: int) -> bool:
        result = self.run(f"kill -0 {pid} 2>/dev/null && echo alive", check=False)
        return "alive" in result.stdout

    def kill_pid(self, pid: int) -> bool:
        result = self.run(f"kill {pid} 2>/dev/null", check=False)
        return result.returncode == 0

    def tail_log(self, log_file: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["zsh", "-c", f"tail -f {log_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def copy_to(self, local_path: Path, remote_path: str) -> None:
        dest = Path(remote_path.replace("~", str(Path.home())))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        src = Path(remote_path.replace("~", str(Path.home())))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)


Backend = RemoteBackend | LocalBackend
