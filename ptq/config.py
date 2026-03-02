from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomllib

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

    def models_for(self, agent: str) -> AgentModels:
        return self.agent_models.get(agent, AgentModels(available=[], default=""))

    def effective_model(self, agent: str, model: str | None = None) -> str:
        if model:
            return model
        am = self.agent_models.get(agent)
        if am:
            return am.default
        return self.default_model


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

    return Config(
        default_agent=defaults.get("agent", "claude"),
        default_model=defaults.get("model", "opus"),
        default_max_turns=defaults.get("max_turns", 100),
        machines=machines_section.get("names", []),
        agent_models=agent_models,
    )


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_TOML)
    return _parse(tomllib.loads(path.read_text()))
