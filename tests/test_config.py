from __future__ import annotations

import textwrap

from ptq.config import Config, _parse, load_config


class TestParse:
    def test_defaults(self):
        config = _parse({})
        assert config.build_env == {"USE_NINJA": "1", "USE_NNPACK": "0"}
        assert config.repos_raw == {}

    def test_build_env(self):
        config = _parse({"build": {"env": {"USE_NINJA": "1", "BUILD_TEST": "0"}}})
        assert config.build_env == {"USE_NINJA": "1", "BUILD_TEST": "0"}

    def test_repositories(self):
        repos = {"example": {"github_repo": "org/example"}}
        assert _parse({"repos": repos}).repos_raw == repos


class TestConfig:
    def test_build_env_prefix(self):
        config = Config(
            build_env={"USE_NINJA": "1", "USE_NNPACK": "0", "BUILD_TEST": "0"}
        )
        prefix = config.build_env_prefix()
        assert "USE_NINJA=1" in prefix
        assert "USE_NNPACK=0" in prefix
        assert "BUILD_TEST=0" in prefix
        assert prefix.endswith(" ")

    def test_build_env_prefix_empty(self):
        assert Config(build_env={}).build_env_prefix() == ""


class TestLoadConfig:
    def test_creates_default_if_missing(self, tmp_path):
        path = tmp_path / "config.toml"
        config = load_config(path)
        assert path.exists()
        assert config.build_env == {"USE_NINJA": "1", "USE_NNPACK": "0"}
        assert "pytorch" in config.repos_raw

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [repos.example]
                github_repo = "org/example"

                [build.env]
                USE_NINJA = "1"
                """
            )
        )
        config = load_config(path)
        assert config.build_env == {"USE_NINJA": "1"}
        assert config.repos_raw["example"]["github_repo"] == "org/example"
