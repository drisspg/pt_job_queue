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

# Optional prompt presets.
# Built-ins always exist, but can be overridden under [prompt_library.builtin.*].
# Add your own under [prompt_library.custom.*].
#
# [prompt_library.builtin.diagnose_and_plan]
# title = "Diagnose And Plan"
# body = "Investigate this issue in diagnosis-and-plan mode."
#
# [prompt_library.custom.my_investigation]
# title = "My Investigation"
# body = "Investigate with my preferred checklist..."

[build.env]
USE_NINJA = "1"
USE_NNPACK = "0"
# Uncomment to skip building NCCL from source (~5 min savings).
# Requires NCCL installed system-wide (e.g. via apt install libnccl-dev).
# USE_SYSTEM_NCCL = "1"
"""


@dataclass
class AgentModels:
    available: list[str]
    default: str


@dataclass(frozen=True)
class PromptPreset:
    key: str
    title: str
    body: str


def _default_prompt_presets() -> list[PromptPreset]:
    return [
        PromptPreset(
            key="repro_only",
            title="Repro Only",
            body="""\
Investigate this issue in reproduction-only mode.

Do not edit product source files.
Do not generate a diff.
Do not rebuild to validate a hypothetical fix.
Do not infer a root cause from static inspection alone.

Your job:
- capture the exact environment: commit, branch, GPU, driver, CUDA, Python env
- find and run the reporter's repro unchanged if possible
- determine whether the issue reproduces on this machine
- if it does not reproduce, stop and report that clearly
- list the most likely reasons this environment differs from the reporter's setup

Hard stops:
- no fix without a failing repro in this environment
- if the tree already contains related edits, report that immediately and do not treat them as proof
- no stash, revert, or rebuild unless the user explicitly asks

Output:
- reproduced / not reproduced
- exact environment
- exact commands run
- likely mismatch explanations
- recommended next experiments

Then wait for user instructions.""",
        ),
        PromptPreset(
            key="diagnose_and_plan",
            title="Diagnose And Plan",
            body="""\
Investigate this issue in diagnosis-and-plan mode.

You may inspect code, compare versions, and create throwaway helpers inside the job directory, but you may not modify product source files or prepare a patch yet.

Use subagents early for parallel exploration. Prefer read-only subagents. Good subagent tasks include:
- finding related kernels or codepaths
- comparing behavior across versions or backends
- locating similar bugs or existing tests
- checking whether the first hypothesis actually matches the runtime evidence

Your goal:
- gather evidence
- produce a minimal repro or a clear non-repro conclusion
- rank the top root-cause hypotheses
- propose the smallest plausible fix plan
- identify unknowns and risks

Rules:
- do not generate a diff
- do not broaden scope mid-run
- do not treat mathematical plausibility or static inspection as sufficient proof
- if runtime evidence and code inspection disagree, trust runtime evidence and report the disagreement

Output:
- repro status
- strongest evidence
- ranked hypotheses with confidence
- likely files involved
- proposed fix plan
- open questions
- recommended next action for user approval

Stop after the plan and wait.""",
        ),
        PromptPreset(
            key="fix_and_verify",
            title="Fix And Verify",
            body="""\
I agree with the reproduction and the proposed solution. Implement and verify it.

Your job:
- write the smallest fix that explains the reproduced failure
- add or update a focused regression test
- demonstrate that the test would fail before the fix
- demonstrate that the same test passes with the fix
- keep scope tight to the approved diagnosis and plan

Rules:
- no speculative cleanup or unrelated refactors
- if the test does not fail before the fix, stop and report that the evidence is insufficient
- if the fix requires a broader change than planned, stop and ask before expanding scope
- prefer the narrowest test that captures the reported failure mode

Output:
- patch summary
- regression test added or updated
- evidence that it fails before the fix
- evidence that it passes after the fix
- any remaining risks or gaps""",
        ),
        PromptPreset(
            key="simplify",
            title="Simplify",
            body="""\
Review the code you just changed and simplify it while preserving exact functionality.

Core principles:
- Never change what the code does — only how it's expressed
- Clarity over brevity — readable, explicit code beats compact or clever code
- Minimal diffs — touch only what needs improving, don't mix cleanup with unrelated changes
- Remove comments that narrate what the code does; keep only non-obvious "why"

Python:
- Inline single-use variables and unnecessary abstractions
- Flatten nesting with early returns and guard clauses
- Prefer match/case over long if/elif chains
- Use X | Y union syntax; never typing.Optional or typing.Union
- Default to @dataclass for simple data containers
- Add or tighten type annotations on function signatures
- Prefer f-strings over .format() or %
- Remove dead code: unused imports, unreachable branches, commented-out code
- Avoid try/except unless the failure mode is expected and common; never swallow exceptions
- No hasattr/getattr for control flow — use explicit attributes, protocols, or ABCs
- Nested functions only when under four lines; extract larger ones to module scope

C++:
- RAII over manual resource management — no raw new/delete
- TORCH_CHECK(condition, message) for user-facing errors — include shapes, dtypes, device in the message
- TORCH_INTERNAL_ASSERT for invariants that indicate bugs, not user mistakes
- Const correctness — mark references, pointers, and methods const where possible
- Prefer range-based for and STL algorithms over raw index loops when they express intent more clearly
- Minimize header includes; prefer forward declarations
- Use auto when the type is obvious from context; spell it out when it aids comprehension
- No clever template metaprogramming unless the surrounding code already establishes the pattern

Anti-patterns to fix:
- Nested ternaries → if/else or switch/match
- Boolean parameters (foo(true, false)) → named constants, enums, or options struct
- isinstance chains → match/case or polymorphism
- Catch-all except Exception → specific exception types or remove try/except
- Magic numbers / string literals → named constants
- Wrapper that only forwards → inline or remove the layer
- Bare assert in C++ → TORCH_CHECK or TORCH_INTERNAL_ASSERT
- printf/std::cout for errors → TORCH_WARN or TORCH_CHECK with context

Flag any change that could alter semantics — when in doubt, don't change it.""",
        ),
    ]


@dataclass
class Config:
    default_agent: str = "claude"
    default_model: str = "opus"
    default_max_turns: int = 100
    machines: list[str] = field(default_factory=list)
    agent_models: dict[str, AgentModels] = field(default_factory=dict)
    prompt_presets: list[PromptPreset] = field(default_factory=_default_prompt_presets)
    build_env: dict[str, str] = field(
        default_factory=lambda: {"USE_NINJA": "1", "USE_NNPACK": "0"}
    )

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

    def prompt_preset(self, name_or_key: str) -> PromptPreset | None:
        needle = name_or_key.strip().lower()
        if not needle:
            return None
        for preset in self.prompt_presets:
            if preset.key.lower() == needle or preset.title.lower() == needle:
                return preset
        return None

    def prompt_preset_choices(self) -> list[str]:
        return [f"{p.key} ({p.title})" for p in self.prompt_presets]


def _parse(data: dict) -> Config:
    defaults = data.get("defaults", {})
    machines_section = data.get("machines", {})
    models_section = data.get("models", {})
    prompt_library_section = data.get("prompt_library", {})

    agent_models: dict[str, AgentModels] = {}
    for agent_name, model_data in models_section.items():
        agent_models[agent_name] = AgentModels(
            available=model_data.get("available", []),
            default=model_data.get("default", ""),
        )

    def _collect_presets(section: dict) -> dict[str, PromptPreset]:
        out: dict[str, PromptPreset] = {}
        for preset_key, preset_data in section.items():
            if not isinstance(preset_data, dict):
                continue
            body = str(preset_data.get("body", "")).strip()
            if not body:
                continue
            normalized_key = str(preset_key).strip().lower().replace("-", "_")
            title = (
                str(preset_data.get("title", "")).strip()
                or normalized_key.replace("_", " ").title()
            )
            out[normalized_key] = PromptPreset(
                key=normalized_key,
                title=title,
                body=body,
            )
        return out

    default_prompt_presets = _default_prompt_presets()
    prompt_presets_by_key = {preset.key: preset for preset in default_prompt_presets}

    # Supported format:
    # [prompt_library.builtin.<key>]  -> overrides built-ins
    # [prompt_library.custom.<key>]   -> adds user presets
    builtins_section = prompt_library_section.get("builtin", {})
    custom_section = prompt_library_section.get("custom", {})

    prompt_presets_by_key.update(_collect_presets(builtins_section))
    prompt_presets_by_key.update(_collect_presets(custom_section))

    default_prompt_keys = [preset.key for preset in default_prompt_presets]
    prompt_presets = [prompt_presets_by_key[key] for key in default_prompt_keys]
    prompt_presets.extend(
        preset
        for key, preset in prompt_presets_by_key.items()
        if key not in default_prompt_keys
    )

    build_section = data.get("build", {})
    build_env = {
        str(k): str(v)
        for k, v in build_section.get(
            "env", {"USE_NINJA": "1", "USE_NNPACK": "0"}
        ).items()
    }

    return Config(
        default_agent=defaults.get("agent", "claude"),
        default_model=defaults.get("model", "opus"),
        default_max_turns=defaults.get("max_turns", 100),
        machines=machines_section.get("names", []),
        agent_models=agent_models,
        prompt_presets=prompt_presets,
        build_env=build_env,
    )


def _extract_hosts(path: Path) -> list[str]:
    if not path.exists():
        return []
    hosts: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("host ") and "*" not in stripped:
            hosts.extend(stripped.split()[1:])
    return hosts


def discover_ssh_hosts() -> list[str]:
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return []
    hosts: list[str] = []
    for line in ssh_config.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("include "):
            pattern = stripped.split(None, 1)[1]
            if pattern.startswith("~"):
                pattern = str(Path.home()) + pattern[1:]
            for included in sorted(Path("/").glob(pattern.lstrip("/"))):
                hosts.extend(_extract_hosts(included))
        elif stripped.lower().startswith("host ") and "*" not in stripped:
            hosts.extend(stripped.split()[1:])
    return hosts


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_TOML)
    return _parse(tomllib.loads(path.read_text()))


_DISCOVER_CMDS: dict[str, list[str]] = {
    "codex": ["codex", "exec", "x", "--model", "__invalid__"],
    "cursor": ["agent", "-p", "x", "--model", "__invalid__", "--force"],
}

_AVAILABLE_RE = re.compile(r"Available models?:\s*(.+)", re.IGNORECASE)

_FALLBACK_MODELS: dict[str, list[str]] = {
    "claude": ["opus", "sonnet", "haiku"],
}

_EXTERNAL_CACHE_FILES: dict[str, tuple[Path, str]] = {
    "codex": (Path.home() / ".codex" / "models_cache.json", "slug"),
}

_DISK_CACHE_PATH = Path.home() / ".ptq" / "discovered_models.json"


def _load_disk_cache() -> dict[str, list[str]]:
    if _DISK_CACHE_PATH.exists():
        try:
            return json.loads(_DISK_CACHE_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_disk_cache(cache: dict[str, list[str]]) -> None:
    _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DISK_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _read_external_cache(agent_name: str) -> list[str]:
    entry = _EXTERNAL_CACHE_FILES.get(agent_name)
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


def cached_models(agent_name: str) -> list[str]:
    return _load_disk_cache().get(agent_name, []) or _FALLBACK_MODELS.get(
        agent_name, []
    )


def discover_models(agent_name: str) -> list[str]:
    models: list[str] = []

    cmd = _DISCOVER_CMDS.get(agent_name)
    if cmd:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            match = _AVAILABLE_RE.search(result.stdout + result.stderr)
            if match:
                models = [m.strip() for m in match.group(1).split(",") if m.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not models:
        models = _read_external_cache(agent_name)

    if not models:
        models = _FALLBACK_MODELS.get(agent_name, [])

    cache = _load_disk_cache()
    cache[agent_name] = models
    _save_disk_cache(cache)

    if models:
        log.debug("discovered %d models for %s", len(models), agent_name)
    return models
