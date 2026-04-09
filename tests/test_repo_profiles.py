from __future__ import annotations

import pytest

from ptq.repo_profiles import (
    _DEFAULT_PROFILES,
    available_repos,
    get_profile,
    load_profiles_from_config,
    reset_cache,
)


class TestGetProfile:
    def test_pytorch(self):
        p = get_profile("pytorch")
        assert p.name == "pytorch"
        assert p.github_repo == "pytorch/pytorch"
        assert p.dir_name == "pytorch"
        assert p.needs_cpp_build is True
        assert p.uses_custom_worktree_tool is True
        assert p.lint_cmd == "spin fixlint"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown repo 'nope'"):
            get_profile("nope")

    def test_all_profiles_have_prompt_templates(self):
        for name, profile in _DEFAULT_PROFILES.items():
            assert profile.prompt_template, f"{name} missing prompt_template"
            assert profile.adhoc_prompt_template, f"{name} missing adhoc_prompt_template"

    def test_profiles_frozen(self):
        p = get_profile("pytorch")
        with pytest.raises(AttributeError):
            p.name = "changed"

    def test_available_repos(self):
        repos = available_repos()
        assert "pytorch" in repos


class TestLoadFromConfig:
    def test_minimal_config(self, tmp_path):
        # Create a minimal prompt file so validation passes
        from ptq.repo_profiles import PROMPTS_DIR

        prompt = PROMPTS_DIR / "investigate_myrepo.md"
        adhoc = PROMPTS_DIR / "adhoc_myrepo.md"
        try:
            prompt.write_text("test")
            adhoc.write_text("test")

            section = {
                "myrepo": {
                    "github_repo": "org/myrepo",
                    "clone_url": "https://github.com/org/myrepo.git",
                    "dir_name": "myrepo",
                    "smoke_test_import": "myrepo",
                    "repro_import_hint": "import myrepo",
                },
            }
            profiles = load_profiles_from_config(section)
            assert "myrepo" in profiles
            p = profiles["myrepo"]
            assert p.github_repo == "org/myrepo"
            assert p.needs_cpp_build is False
            assert p.uses_custom_worktree_tool is False
            assert p.lint_cmd is None
            assert p.prompt_template == "investigate_myrepo.md"
            assert p.adhoc_prompt_template == "adhoc_myrepo.md"
        finally:
            prompt.unlink(missing_ok=True)
            adhoc.unlink(missing_ok=True)

    def test_missing_prompt_raises(self):
        section = {
            "noprompts": {
                "github_repo": "org/noprompts",
                "clone_url": "https://github.com/org/noprompts.git",
                "smoke_test_import": "noprompts",
                "repro_import_hint": "import noprompts",
            },
        }
        with pytest.raises(ValueError, match="not found"):
            load_profiles_from_config(section)

    def test_defaults_override(self, tmp_path):
        """Boolean fields default to False when not specified."""
        from ptq.repo_profiles import PROMPTS_DIR

        prompt = PROMPTS_DIR / "investigate_testdefaults.md"
        adhoc = PROMPTS_DIR / "adhoc_testdefaults.md"
        try:
            prompt.write_text("test")
            adhoc.write_text("test")
            section = {
                "testdefaults": {
                    "github_repo": "org/testdefaults",
                    "clone_url": "https://github.com/org/testdefaults.git",
                    "smoke_test_import": "testdefaults",
                    "repro_import_hint": "import testdefaults",
                    "uses_custom_worktree_tool": True,
                    "needs_cpp_build": True,
                    "lint_cmd": "make lint",
                },
            }
            profiles = load_profiles_from_config(section)
            p = profiles["testdefaults"]
            assert p.uses_custom_worktree_tool is True
            assert p.needs_cpp_build is True
            assert p.lint_cmd == "make lint"
        finally:
            prompt.unlink(missing_ok=True)
            adhoc.unlink(missing_ok=True)
