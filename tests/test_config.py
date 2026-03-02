from __future__ import annotations

import tempfile
from pathlib import Path

from ptq.config import Config, load_config


class TestLoadConfig:
    def test_creates_default_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            cfg = load_config(path)
            assert path.exists()
            assert cfg.default_agent == "claude"
            assert cfg.default_max_turns == 100
            assert cfg.agent_models["claude"].default == "opus"
            assert cfg.agent_models["claude"].available == []

    def test_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[defaults]\nagent = "codex"\nmodel = "o3"\nmax_turns = 50\n'
                "[machines]\nnames = ['gpu-dev']\n"
                '[models.codex]\navailable = ["o3", "o4-mini"]\ndefault = "o3"\n'
            )
            cfg = load_config(path)
            assert cfg.default_agent == "codex"
            assert cfg.default_model == "o3"
            assert cfg.default_max_turns == 50
            assert cfg.machines == ["gpu-dev"]
            assert cfg.agent_models["codex"].available == ["o3", "o4-mini"]

    def test_missing_sections_use_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("")
            cfg = load_config(path)
            assert cfg.default_agent == "claude"
            assert cfg.machines == []
            assert cfg.agent_models == {}


class TestEffectiveModel:
    def test_explicit_model_wins(self):
        cfg = Config()
        assert cfg.effective_model("claude", "haiku") == "haiku"

    def test_agent_default(self):
        from ptq.config import AgentModels

        cfg = Config(
            agent_models={
                "codex": AgentModels(available=["o3", "o4-mini"], default="o3")
            }
        )
        assert cfg.effective_model("codex", None) == "o3"

    def test_falls_back_to_global(self):
        cfg = Config(default_model="sonnet")
        assert cfg.effective_model("unknown", None) == "sonnet"


class TestModelsFor:
    def test_known_agent(self):
        from ptq.config import AgentModels

        cfg = Config(
            agent_models={"claude": AgentModels(available=["opus"], default="opus")}
        )
        am = cfg.models_for("claude")
        assert am.available == ["opus"]

    def test_unknown_agent(self):
        cfg = Config()
        am = cfg.models_for("unknown")
        assert am.available == []
        assert am.default == ""
