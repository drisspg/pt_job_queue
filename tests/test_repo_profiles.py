from __future__ import annotations

import pytest

from ptq.repo_profiles import available_repos, get_profile, load_profiles_from_config


class TestGetProfile:
    def test_pytorch(self):
        profile = get_profile("pytorch")
        assert profile.name == "pytorch"
        assert profile.github_repo == "pytorch/pytorch"
        assert profile.dir_name == "pytorch"
        assert profile.needs_cpp_build is True
        assert profile.uses_custom_worktree_tool is True
        assert profile.lint_cmd == "spin fixlint"

    def test_torchtitan(self):
        profile = get_profile("torchtitan")
        assert profile.name == "torchtitan"
        assert profile.github_repo == "pytorch/torchtitan"
        assert profile.dir_name == "torchtitan"
        assert profile.needs_cpp_build is False
        assert profile.uses_custom_worktree_tool is False
        assert profile.lint_cmd is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown repo 'nope'"):
            get_profile("nope")

    def test_profiles_frozen(self):
        profile = get_profile("pytorch")
        with pytest.raises(AttributeError):
            profile.name = "changed"

    def test_available_repos(self):
        repos = available_repos()
        assert "pytorch" in repos
        assert "torchtitan" in repos


class TestLoadFromConfig:
    def test_minimal_config(self):
        profiles = load_profiles_from_config(
            {
                "myrepo": {
                    "github_repo": "org/myrepo",
                    "clone_url": "https://github.com/org/myrepo.git",
                    "dir_name": "myrepo",
                    "smoke_test_import": "myrepo",
                }
            }
        )
        profile = profiles["myrepo"]
        assert profile.github_repo == "org/myrepo"
        assert profile.needs_cpp_build is False
        assert profile.uses_custom_worktree_tool is False
        assert profile.lint_cmd is None

    def test_optional_fields(self):
        profiles = load_profiles_from_config(
            {
                "custom": {
                    "github_repo": "org/custom",
                    "clone_url": "https://github.com/org/custom.git",
                    "smoke_test_import": "custom",
                    "uses_custom_worktree_tool": True,
                    "needs_cpp_build": True,
                    "lint_cmd": "make lint",
                }
            }
        )
        profile = profiles["custom"]
        assert profile.uses_custom_worktree_tool is True
        assert profile.needs_cpp_build is True
        assert profile.lint_cmd == "make lint"
