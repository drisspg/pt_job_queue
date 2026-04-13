from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass(frozen=True)
class RepoProfile:
    name: str
    github_repo: str
    clone_url: str
    dir_name: str
    smoke_test_import: str
    repro_import_hint: str
    uses_custom_worktree_tool: bool
    needs_cpp_build: bool
    lint_cmd: str | None
    prompt_template: str
    adhoc_prompt_template: str


def _resolve_prompt_templates(name: str) -> tuple[str, str]:
    """Derive prompt filenames by convention, with pytorch backward compat."""
    if name == "pytorch":
        return "investigate.md", "adhoc.md"
    return f"investigate_{name}.md", f"adhoc_{name}.md"


def _validate_prompt_templates(profile: RepoProfile) -> None:
    for attr in ("prompt_template", "adhoc_prompt_template"):
        filename = getattr(profile, attr)
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise ValueError(
                f"Prompt template '{filename}' not found at {path}. "
                f"Create it to use the '{profile.name}' repo profile."
            )


# Built-in defaults used when config has no [repos] section.
_DEFAULT_PROFILES: dict[str, RepoProfile] = {
    "pytorch": RepoProfile(
        name="pytorch",
        github_repo="pytorch/pytorch",
        clone_url="https://github.com/pytorch/pytorch.git",
        dir_name="pytorch",
        smoke_test_import="torch",
        repro_import_hint="import torch",
        uses_custom_worktree_tool=True,
        needs_cpp_build=True,
        lint_cmd="spin fixlint",
        prompt_template="investigate.md",
        adhoc_prompt_template="adhoc.md",
    ),
    "torchtitan": RepoProfile(
        name="torchtitan",
        github_repo="pytorch/torchtitan",
        clone_url="https://github.com/pytorch/torchtitan.git",
        dir_name="torchtitan",
        smoke_test_import="torchtitan",
        repro_import_hint="import torchtitan",
        uses_custom_worktree_tool=False,
        needs_cpp_build=False,
        lint_cmd=None,
        prompt_template="investigate_torchtitan.md",
        adhoc_prompt_template="adhoc_torchtitan.md",
    ),
}


def load_profiles_from_config(repos_section: dict) -> dict[str, RepoProfile]:
    """Parse [repos.*] TOML sections into RepoProfile instances."""
    profiles: dict[str, RepoProfile] = {}
    for name, data in repos_section.items():
        if not isinstance(data, dict):
            continue
        investigate, adhoc = _resolve_prompt_templates(name)
        profiles[name] = RepoProfile(
            name=name,
            github_repo=data["github_repo"],
            clone_url=data["clone_url"],
            dir_name=data.get("dir_name", name),
            smoke_test_import=data["smoke_test_import"],
            repro_import_hint=data["repro_import_hint"],
            uses_custom_worktree_tool=data.get("uses_custom_worktree_tool", False),
            needs_cpp_build=data.get("needs_cpp_build", False),
            lint_cmd=data.get("lint_cmd"),
            prompt_template=data.get("prompt_template", investigate),
            adhoc_prompt_template=data.get("adhoc_prompt_template", adhoc),
        )
    for profile in profiles.values():
        _validate_prompt_templates(profile)
    return profiles


_profiles_cache: dict[str, RepoProfile] | None = None


def _loaded_profiles() -> dict[str, RepoProfile]:
    global _profiles_cache
    if _profiles_cache is None:
        from ptq.config import load_config

        cfg = load_config()
        if cfg.repos_raw:
            _profiles_cache = load_profiles_from_config(cfg.repos_raw)
        else:
            _profiles_cache = dict(_DEFAULT_PROFILES)
    return _profiles_cache


def get_profile(name: str) -> RepoProfile:
    profiles = _loaded_profiles()
    profile = profiles.get(name)
    if profile is None:
        raise ValueError(
            f"Unknown repo '{name}'. Available: {', '.join(profiles)}"
        )
    return profile


def available_repos() -> list[str]:
    return list(_loaded_profiles())


def reset_cache() -> None:
    """Clear cached profiles (for testing)."""
    global _profiles_cache
    _profiles_cache = None
