from __future__ import annotations

from rich.console import Console

from ptq.ssh import Backend, RemoteBackend

console = Console()


def setup_workspace(backend: Backend, *, build: bool = False) -> None:
    workspace = backend.workspace

    console.print(f"[bold]Setting up workspace at {workspace}[/bold]")

    if isinstance(backend, RemoteBackend):
        _install_uv_remote(backend)

    console.print("Creating workspace directories...")
    backend.run(f"mkdir -p {workspace}/jobs {workspace}/scripts")

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

    if build:
        build_pytorch(backend)

    console.print("Deploying helper scripts...")
    deploy_scripts(backend)

    console.print("[bold green]Workspace setup complete.[/bold green]")


def _clone_pytorch(backend: Backend, workspace: str) -> None:
    existing = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if existing.returncode == 0:
        console.print("PyTorch checkout already exists, pulling latest...")
        backend.run(f"cd {workspace}/pytorch && git pull", stream=True)
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


def build_pytorch(backend: Backend) -> None:
    workspace = backend.workspace
    console.print(
        "[bold]Building PyTorch from source (this may take a while)...[/bold]"
    )
    result = backend.run(
        f"cd {workspace}/pytorch && USE_NINJA=1 {workspace}/.venv/bin/pip install -v -e .",
        check=False,
        stream=True,
    )
    if result.returncode != 0:
        raise SystemExit("Build failed.")

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


def _install_uv_remote(backend: RemoteBackend) -> None:
    check = backend.run("which uv", check=False)
    if check.returncode == 0:
        console.print("uv already installed.")
        return
    console.print("Installing uv on remote...")
    backend.run("curl -LsSf https://astral.sh/uv/install.sh | sh")
