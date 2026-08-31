from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomllib

CONFIG_PATH = Path.home() / ".ptq" / "config.toml"

_DEFAULT_TOML = """\
[repos.pytorch]
github_repo = "pytorch/pytorch"
clone_url = "https://github.com/pytorch/pytorch.git"
dir_name = "pytorch"
smoke_test_import = "torch"
uses_custom_worktree_tool = true
needs_cpp_build = true
lint_cmd = "spin fixlint"

[repos.torchtitan]
github_repo = "pytorch/torchtitan"
clone_url = "https://github.com/pytorch/torchtitan.git"
dir_name = "torchtitan"
smoke_test_import = "torchtitan"

[build.env]
USE_NINJA = "1"
USE_NNPACK = "0"
# Uncomment to skip building NCCL from source (~5 min savings).
# Requires NCCL installed system-wide (e.g. via apt install libnccl-dev).
# USE_SYSTEM_NCCL = "1"
"""


@dataclass
class Config:
    build_env: dict[str, str] = field(
        default_factory=lambda: {
            "USE_NINJA": "1",
            "USE_NNPACK": "0",
            "BUILD_TEST": "0",
        }
    )
    repos_raw: dict = field(default_factory=dict)

    def build_env_prefix(self) -> str:
        if not self.build_env:
            return ""
        return " ".join(f"{key}={value}" for key, value in self.build_env.items()) + " "


def _parse(data: dict) -> Config:
    build_env = {
        str(key): str(value)
        for key, value in data.get("build", {})
        .get("env", {"USE_NINJA": "1", "USE_NNPACK": "0"})
        .items()
    }
    return Config(build_env=build_env, repos_raw=data.get("repos", {}))


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_TOML)
    return _parse(tomllib.loads(path.read_text()))
