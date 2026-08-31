from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoProfile:
    name: str
    github_repo: str
    clone_url: str
    dir_name: str
    smoke_test_import: str
    uses_custom_worktree_tool: bool
    needs_cpp_build: bool
    lint_cmd: str | None


# Built-in defaults used when config has no [repos] section.
_DEFAULT_PROFILES: dict[str, RepoProfile] = {
    "pytorch": RepoProfile(
        name="pytorch",
        github_repo="pytorch/pytorch",
        clone_url="https://github.com/pytorch/pytorch.git",
        dir_name="pytorch",
        smoke_test_import="torch",
        uses_custom_worktree_tool=True,
        needs_cpp_build=True,
        lint_cmd="spin fixlint",
    ),
    "torchtitan": RepoProfile(
        name="torchtitan",
        github_repo="pytorch/torchtitan",
        clone_url="https://github.com/pytorch/torchtitan.git",
        dir_name="torchtitan",
        smoke_test_import="torchtitan",
        uses_custom_worktree_tool=False,
        needs_cpp_build=False,
        lint_cmd=None,
    ),
}


def load_profiles_from_config(repos_section: dict) -> dict[str, RepoProfile]:
    """Parse [repos.*] TOML sections into RepoProfile instances."""
    profiles: dict[str, RepoProfile] = {}
    for name, data in repos_section.items():
        if not isinstance(data, dict):
            continue
        profiles[name] = RepoProfile(
            name=name,
            github_repo=data["github_repo"],
            clone_url=data["clone_url"],
            dir_name=data.get("dir_name", name),
            smoke_test_import=data["smoke_test_import"],
            uses_custom_worktree_tool=data.get("uses_custom_worktree_tool", False),
            needs_cpp_build=data.get("needs_cpp_build", False),
            lint_cmd=data.get("lint_cmd"),
        )
    return profiles


_profiles_cache: dict[str, RepoProfile] | None = None


def _loaded_profiles() -> dict[str, RepoProfile]:
    global _profiles_cache
    if _profiles_cache is None:
        from ptq.config import load_config

        cfg = load_config()
        repos_raw = cfg.repos_raw if isinstance(cfg.repos_raw, dict) else {}
        loaded = load_profiles_from_config(repos_raw) if repos_raw else {}
        _profiles_cache = loaded or dict(_DEFAULT_PROFILES)
    return _profiles_cache


def get_profile(name: str) -> RepoProfile:
    profiles = _loaded_profiles()
    profile = profiles.get(name)
    if profile is None:
        raise ValueError(f"Unknown repo '{name}'. Available: {', '.join(profiles)}")
    return profile


def available_repos() -> list[str]:
    return list(_loaded_profiles())


def reset_cache() -> None:
    """Clear cached profiles (for testing)."""
    global _profiles_cache
    _profiles_cache = None
