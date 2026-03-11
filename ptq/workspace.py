from __future__ import annotations

import re

from rich.console import Console

from ptq.ssh import Backend, RemoteBackend

_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")

_SUPPORTED_CUDA = {
    (12, 4): "cu124",
    (12, 6): "cu126",
    (12, 8): "cu128",
    (13, 0): "cu130",
}


def detect_cuda_version(backend: Backend) -> str:
    result = backend.run("nvidia-smi", check=False)
    if result.returncode != 0:
        raise SystemExit("nvidia-smi not found or failed.")
    m = _CUDA_VERSION_RE.search(result.stdout)
    if not m:
        raise SystemExit("Could not parse CUDA version from nvidia-smi output.")
    major, minor = int(m.group(1)), int(m.group(2))
    if major < 12:
        raise SystemExit(f"CUDA {major}.{minor} is too old (need >= 12.4).")
    best = None
    for (sup_major, sup_minor), tag in sorted(_SUPPORTED_CUDA.items()):
        if (major, minor) >= (sup_major, sup_minor):
            best = tag
    if best is None:
        raise SystemExit(f"CUDA {major}.{minor} is too old (need >= 12.4).")
    return best


console = Console()


def setup_workspace(
    backend: Backend,
    *,
    build: bool = False,
    re_cc_jobs: int = 0,
    build_env_prefix: str = "USE_NINJA=1 ",
) -> None:
    workspace = backend.workspace

    console.print(f"[bold]Setting up workspace at {workspace}[/bold]")

    if isinstance(backend, RemoteBackend):
        _install_uv_remote(backend)
        _ensure_rsync(backend)

    console.print("Creating workspace directories...")
    backend.run(f"mkdir -p {workspace}/jobs {workspace}/scripts")

    _ensure_ccache_config(backend)
    _clone_pytorch(backend, workspace)

    console.print("Installing Python 3.12 via uv...")
    backend.run("uv python install 3.12", check=False)

    console.print("Creating venv...")
    backend.run(f"cd {workspace} && uv venv --python 3.12", check=False)

    console.print("Installing build dependencies...")
    result = backend.run(
        f"cd {workspace} && uv pip install --python .venv/bin/python "
        f"-r {workspace}/pytorch/requirements-build.txt",
        check=False,
        stream=True,
    )
    if result.returncode != 0:
        raise SystemExit("Installing build dependencies failed.")

    _install_triton(backend, workspace)

    if build:
        build_pytorch(backend, re_cc_jobs=re_cc_jobs, build_env_prefix=build_env_prefix)

    console.print("Deploying helper scripts...")
    deploy_scripts(backend)

    console.print("[bold green]Workspace setup complete.[/bold green]")


def _clone_pytorch(backend: Backend, workspace: str) -> None:
    existing = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if existing.returncode == 0:
        console.print("PyTorch checkout already exists, resetting to latest...")
        backend.run(
            f"cd {workspace}/pytorch && git fetch origin && git reset --hard origin/main",
            stream=True,
        )
        backend.run(
            f"cd {workspace}/pytorch && git submodule sync && git submodule update --init --recursive --progress",
            stream=True,
        )
        return

    console.print("Cloning pytorch (full clone with submodules)...")
    backend.run(
        f"git clone --progress https://github.com/pytorch/pytorch.git {workspace}/pytorch",
        stream=True,
    )
    backend.run(
        f"cd {workspace}/pytorch && git submodule update --init --recursive --progress",
        stream=True,
    )


def _install_triton(backend: Backend, workspace: str) -> None:
    console.print("Installing Triton...")
    r = backend.run(
        f"cd {workspace}/pytorch && PATH={workspace}/.venv/bin:$PATH make triton",
        check=False,
        stream=True,
    )
    if r.returncode != 0:
        console.print("[yellow]Triton install failed (non-fatal).[/yellow]")


def build_pytorch(
    backend: Backend, *, re_cc_jobs: int = 0, build_env_prefix: str = "USE_NINJA=1 "
) -> None:
    workspace = backend.workspace
    console.print(
        "[bold]Building PyTorch from source (this may take a while)...[/bold]"
    )

    # Full nuke before build: mirrors Driss's local workflow and avoids stale
    # build graphs after upstream renames/deletes.
    console.print("Nuking local PyTorch checkout artifacts before build...")
    backend.run(
        f"cd {workspace}/pytorch && git clean -dfx && git submodule sync && git submodule update --init --recursive",
        check=False,
        stream=True,
    )

    pip_cmd = f"{workspace}/.venv/bin/pip install -v -e ."
    if re_cc_jobs > 0:
        pip_cmd = f"re-cc -- {workspace}/.venv/bin/pip install -v -e ."
        build_env_prefix = f"MAX_JOBS={re_cc_jobs} {build_env_prefix}"
        console.print(f"[bold]Using re-cc with MAX_JOBS={re_cc_jobs}[/bold]")
        backend.run(f"echo {re_cc_jobs} > {workspace}/.re-cc-config")

    result = backend.run(
        f"cd {workspace}/pytorch && {build_env_prefix}{pip_cmd}",
        check=False,
        stream=True,
    )
    if result.returncode != 0:
        raise SystemExit("Build failed.")

    _install_triton(backend, workspace)

    console.print("Running smoke test...")
    smoke = backend.run(
        f'cd /tmp && {workspace}/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
    )
    console.print(f"[green]Smoke test: {smoke.stdout.strip()}[/green]")
    console.print("[bold green]Build complete.[/bold green]")


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


_CCACHE_CONF = """\
max_size = 25G
base_dir = {home}
"""


def _ensure_ccache_config(backend: Backend) -> None:
    result = backend.run("which ccache", check=False)
    if result.returncode != 0:
        console.print("[yellow]ccache not found, skipping config.[/yellow]")
        return

    conf_dir = "~/.config/ccache"
    conf_file = f"{conf_dir}/ccache.conf"
    existing = backend.run(f"cat {conf_file}", check=False)
    if existing.returncode == 0 and "base_dir" in existing.stdout:
        console.print("ccache config already has base_dir, skipping.")
        return

    home = backend.run("echo $HOME", check=False).stdout.strip()
    conf = _CCACHE_CONF.format(home=home)
    backend.run(f"mkdir -p {conf_dir}")
    backend.run(f"cat > {conf_file} << 'CCACHE_EOF'\n{conf}CCACHE_EOF")
    console.print(f"Configured ccache with base_dir={home}")


def _ensure_rsync(backend: Backend) -> None:
    if backend.run("which rsync", check=False).returncode == 0:
        return
    console.print("Installing rsync...")
    if backend.run("sudo apt-get install -y rsync", check=False).returncode == 0:
        return
    if backend.run("sudo yum install -y rsync", check=False).returncode == 0:
        return
    console.print(
        "[yellow]Could not install rsync — fast-path venv clone will be disabled.[/yellow]"
    )


def _install_uv_remote(backend: RemoteBackend) -> None:
    check = backend.run("which uv", check=False)
    if check.returncode == 0:
        console.print("uv already installed.")
        return
    console.print("Installing uv on remote...")
    backend.run("curl -LsSf https://astral.sh/uv/install.sh | sh")
