from __future__ import annotations

import re
import subprocess

from rich.console import Console

from ptq.ssh import Backend, LocalBackend, RemoteBackend

console = Console()

CUDA_INDEX_MAP = {
    "12.4": "cu124",
    "12.6": "cu126",
    "12.8": "cu128",
    "13.0": "cu130",
}

DEFAULT_CUDA_TAG = "cu130"


def detect_cuda_version(backend: Backend) -> str:
    result = backend.run("nvidia-smi", check=False)
    if result.returncode != 0:
        raise SystemExit("nvidia-smi not found on target machine. Use --cuda to specify CUDA version.")
    match = re.search(r"CUDA Version:\s+(\d+\.\d+)", result.stdout)
    if not match:
        raise SystemExit("Could not parse CUDA version from nvidia-smi. Use --cuda to specify.")
    driver_version = tuple(int(x) for x in match.group(1).split("."))
    best = None
    for version_str, tag in CUDA_INDEX_MAP.items():
        toolkit_version = tuple(int(x) for x in version_str.split("."))
        if toolkit_version <= driver_version and (best is None or toolkit_version > best[0]):
            best = (toolkit_version, tag)
    if best is None:
        raise SystemExit(f"CUDA driver {match.group(1)} is too old. Minimum supported: {min(CUDA_INDEX_MAP)}.")
    return best[1]


def setup_workspace(backend: Backend, cuda_tag: str | None = None, cpu: bool = False) -> None:
    workspace = backend.workspace

    if cpu:
        cuda_tag = "cpu"
    elif cuda_tag is None:
        cuda_tag = detect_cuda_version(backend)
    index_url = f"https://download.pytorch.org/whl/nightly/{cuda_tag}"

    console.print(f"[bold]Setting up workspace at {workspace}[/bold]")

    if isinstance(backend, RemoteBackend):
        _install_uv_remote(backend)

    console.print("Creating workspace directories...")
    backend.run(f"mkdir -p {workspace}/jobs {workspace}/scripts")

    console.print("Creating venv...")
    backend.run(f"cd {workspace} && uv venv --python 3.12", check=False)

    console.print(f"Installing PyTorch nightly ({cuda_tag})...")
    result = backend.run(
        f"cd {workspace} && uv pip install --python .venv/bin/python --pre torch --index-url {index_url}",
        check=False,
    )
    if result.returncode != 0:
        console.print(f"[red]pip install failed:[/red]\n{result.stderr}")
        raise SystemExit(1)
    console.print(result.stdout or "done")

    console.print("Resolving nightly commit hash...")
    nightly_hash = backend.run(
        f"{workspace}/.venv/bin/python -c \"import torch; print(torch.version.git_version)\"",
    ).stdout.strip()
    console.print(f"Nightly git version: {nightly_hash}")

    main_hash = _resolve_main_hash(nightly_hash)
    console.print(f"Main commit: {main_hash}")

    console.print("Cloning pytorch source...")
    backend.run(f"rm -rf {workspace}/pytorch", check=False)
    backend.run(
        f"git clone --depth 1 https://github.com/pytorch/pytorch.git {workspace}/pytorch"
    )
    backend.run(
        f"cd {workspace}/pytorch && git fetch --depth 1 origin {main_hash} && git checkout {main_hash}"
    )

    console.print("Deploying helper scripts...")
    deploy_scripts(backend)

    console.print("Running smoke test...")
    smoke = backend.run(
        f"{workspace}/.venv/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"",
    )
    console.print(f"[green]Smoke test: {smoke.stdout.strip()}[/green]")
    console.print("[bold green]Workspace setup complete.[/bold green]")


def deploy_scripts(backend: Backend) -> None:
    from pathlib import Path

    workspace = backend.workspace
    scripts_dir = Path(__file__).parent.parent / "scripts"
    backend.run(f"mkdir -p {workspace}/scripts")

    if isinstance(backend, RemoteBackend):
        for script in scripts_dir.glob("*.sh"):
            backend.copy_to(script, f"{workspace}/scripts/{script.name}")
    else:
        import shutil

        dest_dir = Path(workspace.replace("~", str(Path.home()))) / "scripts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for script in scripts_dir.glob("*.sh"):
            shutil.copy2(script, dest_dir / script.name)

    backend.run(f"chmod +x {workspace}/scripts/*.sh")


def _install_uv_remote(backend: RemoteBackend) -> None:
    check = backend.run("which uv", check=False)
    if check.returncode == 0:
        console.print("uv already installed.")
        return
    console.print("Installing uv on remote...")
    backend.run("curl -LsSf https://astral.sh/uv/install.sh | sh")


def _resolve_main_hash(nightly_hash: str) -> str:
    result = subprocess.run(
        ["gh", "api", f"repos/pytorch/pytorch/commits/{nightly_hash}", "--jq", ".commit.message"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        msg = result.stdout.strip()
        match = re.search(r"([0-9a-f]{40})", msg)
        if match:
            return match.group(1)
    return nightly_hash
