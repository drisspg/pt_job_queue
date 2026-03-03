from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

log = logging.getLogger("ptq.config")

CONFIG_PATH = Path.home() / ".ptq" / "config.toml"

_DEFAULT_TOML = """\
[defaults]
agent = "claude"
max_turns = 100

[machines]
# Add your machines here. The web UI will show these as a dropdown.
# names = ["gpu-dev", "gpu-prod"]
names = []

# Per-agent model defaults. "default" is used when --model is omitted.
# Add "available" to get a dropdown in the web UI instead of freeform text:
#   available = ["opus", "sonnet", "haiku"]

[models.claude]
default = "opus"

[models.codex]
default = "o3"

[models.cursor]
default = "auto"

[build.env]
USE_NINJA = "1"
USE_NNPACK = "0"
"""


@dataclass
class AgentModels:
    available: list[str]
    default: str


@dataclass
class Config:
    default_agent: str = "claude"
    default_model: str = "opus"
    default_max_turns: int = 100
    machines: list[str] = field(default_factory=list)
    agent_models: dict[str, AgentModels] = field(default_factory=dict)
    build_env: dict[str, str] = field(default_factory=lambda: {"USE_NINJA": "1"})

    def models_for(self, agent: str) -> AgentModels:
        return self.agent_models.get(agent, AgentModels(available=[], default=""))

    def effective_model(self, agent: str, model: str | None = None) -> str:
        if model:
            return model
        am = self.agent_models.get(agent)
        if am:
            return am.default
        return self.default_model

    def build_env_prefix(self) -> str:
        if not self.build_env:
            return ""
        return " ".join(f"{k}={v}" for k, v in self.build_env.items()) + " "


def _parse(data: dict) -> Config:
    defaults = data.get("defaults", {})
    machines_section = data.get("machines", {})
    models_section = data.get("models", {})

    agent_models: dict[str, AgentModels] = {}
    for agent_name, model_data in models_section.items():
        agent_models[agent_name] = AgentModels(
            available=model_data.get("available", []),
            default=model_data.get("default", ""),
        )

    build_section = data.get("build", {})
    build_env = {
        str(k): str(v) for k, v in build_section.get("env", {"USE_NINJA": "1"}).items()
    }

    return Config(
        default_agent=defaults.get("agent", "claude"),
        default_model=defaults.get("model", "opus"),
        default_max_turns=defaults.get("max_turns", 100),
        machines=machines_section.get("names", []),
        agent_models=agent_models,
        build_env=build_env,
    )


def discover_ssh_hosts() -> list[str]:
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return []
    hosts: list[str] = []
    for line in ssh_config.read_text().splitlines():
        line = line.strip()
        if line.lower().startswith("host ") and "*" not in line:
            hosts.extend(line.split()[1:])
    return hosts


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_TOML)
    return _parse(tomllib.loads(path.read_text()))


_DISCOVER_CMDS: dict[str, list[str]] = {
    "claude": ["claude", "-p", "x", "--model", "__invalid__", "--max-turns", "0"],
    "codex": ["codex", "exec", "x", "--model", "__invalid__"],
    "cursor": ["agent", "-p", "x", "--model", "__invalid__", "--force"],
}

_AVAILABLE_RE = re.compile(r"Available models?:\s*(.+)", re.IGNORECASE)

_MODEL_CACHE_FILES: dict[str, tuple[Path, str]] = {
    "codex": (Path.home() / ".codex" / "models_cache.json", "slug"),
}

_discovered_cache: dict[str, list[str]] = {}


def _read_cache_file(agent_name: str) -> list[str]:
    entry = _MODEL_CACHE_FILES.get(agent_name)
    if not entry:
        return []
    path, key = entry
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [
            m[key] for m in data.get("models", []) if isinstance(m, dict) and key in m
        ]
    except (json.JSONDecodeError, KeyError):
        return []


def discover_models(agent_name: str) -> list[str]:
    if agent_name in _discovered_cache:
        return _discovered_cache[agent_name]

    models: list[str] = []

    cmd = _DISCOVER_CMDS.get(agent_name)
    if cmd:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            match = _AVAILABLE_RE.search(result.stdout + result.stderr)
            if match:
                models = [m.strip() for m in match.group(1).split(",") if m.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not models:
        models = _read_cache_file(agent_name)

    _discovered_cache[agent_name] = models
    if models:
        log.debug("discovered %d models for %s", len(models), agent_name)
    return models
