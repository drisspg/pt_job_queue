from __future__ import annotations

import textwrap

from ptq.config import AgentModels, Config, _parse, load_config


class TestParse:
    def test_defaults(self):
        cfg = _parse({})
        assert cfg.default_agent == "claude"
        assert cfg.default_max_turns == 100
        assert [preset.key for preset in cfg.prompt_presets] == [
            "repro_only",
            "diagnose_and_plan",
            "fix_and_verify",
        ]

    def test_custom_agent(self):
        cfg = _parse({"defaults": {"agent": "codex"}})
        assert cfg.default_agent == "codex"

    def test_machines(self):
        cfg = _parse({"machines": {"names": ["gpu-dev", "gpu-prod"]}})
        assert cfg.machines == ["gpu-dev", "gpu-prod"]

    def test_models(self):
        cfg = _parse(
            {
                "models": {
                    "claude": {"default": "opus", "available": ["opus", "sonnet"]},
                    "codex": {"default": "o3"},
                }
            }
        )
        assert cfg.agent_models["claude"].default == "opus"
        assert cfg.agent_models["claude"].available == ["opus", "sonnet"]
        assert cfg.agent_models["codex"].default == "o3"
        assert cfg.agent_models["codex"].available == []

    def test_build_env(self):
        cfg = _parse({"build": {"env": {"USE_NINJA": "1", "USE_NNPACK": "0"}}})
        assert cfg.build_env == {"USE_NINJA": "1", "USE_NNPACK": "0"}

    def test_build_env_defaults(self):
        cfg = _parse({})
        assert cfg.build_env == {"USE_NINJA": "1"}

    def test_prompt_library_overrides_and_appends(self):
        cfg = _parse(
            {
                "prompt_library": {
                    "diagnose_and_plan": {
                        "title": "Diagnosis First",
                        "body": "Plan before patching.",
                    },
                    "custom_triage": {
                        "title": "Custom Triage",
                        "body": "Collect evidence first.",
                    },
                }
            }
        )
        assert cfg.prompt_presets[1].title == "Diagnosis First"
        assert cfg.prompt_presets[1].body == "Plan before patching."
        assert cfg.prompt_presets[-1].key == "custom_triage"
        assert cfg.prompt_presets[-1].title == "Custom Triage"


class TestConfig:
    def test_effective_model_with_override(self):
        cfg = Config()
        assert cfg.effective_model("claude", "sonnet") == "sonnet"

    def test_effective_model_default(self):
        cfg = Config(agent_models={"claude": AgentModels(available=[], default="opus")})
        assert cfg.effective_model("claude") == "opus"

    def test_effective_model_fallback(self):
        cfg = Config(default_model="haiku")
        assert cfg.effective_model("unknown") == "haiku"

    def test_build_env_prefix(self):
        cfg = Config(build_env={"USE_NINJA": "1", "USE_NNPACK": "0"})
        prefix = cfg.build_env_prefix()
        assert "USE_NINJA=1" in prefix
        assert "USE_NNPACK=0" in prefix
        assert prefix.endswith(" ")

    def test_build_env_prefix_empty(self):
        cfg = Config(build_env={})
        assert cfg.build_env_prefix() == ""

    def test_models_for_known(self):
        cfg = Config(
            agent_models={"claude": AgentModels(available=["a", "b"], default="a")}
        )
        am = cfg.models_for("claude")
        assert am.available == ["a", "b"]
        assert am.default == "a"

    def test_models_for_unknown(self):
        cfg = Config()
        am = cfg.models_for("unknown")
        assert am.available == []
        assert am.default == ""


class TestLoadConfig:
    def test_creates_default_if_missing(self, tmp_path):
        path = tmp_path / "config.toml"
        cfg = load_config(path)
        assert path.exists()
        assert cfg.default_agent == "claude"

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            textwrap.dedent("""\
            [defaults]
            agent = "codex"
            max_turns = 50

            [machines]
            names = ["box-a"]

            [models.codex]
            default = "o3"
            """)
        )
        cfg = load_config(path)
        assert cfg.default_agent == "codex"
        assert cfg.default_max_turns == 50
        assert cfg.machines == ["box-a"]
        assert cfg.agent_models["codex"].default == "o3"
