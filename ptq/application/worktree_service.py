from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from subprocess import CompletedProcess

from ptq.domain.models import PtqError
from ptq.ssh import Backend

log = logging.getLogger("ptq.worktree")

ProgressCallback = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    pass


@contextmanager
def _timed(label: str, progress: ProgressCallback):
    t0 = time.monotonic()
    yield
    progress(f"  {label}: {time.monotonic() - t0:.1f}s")


def _chain_result(
    result: CompletedProcess[str],
    next_step: Callable[[], CompletedProcess[str]],
) -> CompletedProcess[str]:
    if result.returncode != 0:
        return result
    return next_step()


def validate_workspace(backend: Backend, workspace: str) -> None:
    result = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if result.returncode != 0:
        raise PtqError(
            f"Workspace broken: {workspace}/pytorch/.git missing. Re-run: ptq setup"
        )


def _try_clone_base_venv(
    backend: Backend,
    job_dir: str,
    worktree_path: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> bool:
    """Clone base workspace venv + source artifacts instead of rebuilding.

    Copies the venv (hardlinks), rsyncs gitignored source artifacts
    (.so, generated .py) into the worktree, and rewrites editable-install
    paths.  Skips the cmake build dir — if the agent needs to rebuild C++,
    ccache handles it (~3 min with warm cache).
    Falls back to the slow editable-install if anything fails.
    """
    workspace = backend.workspace
    base_venv = f"{workspace}/.venv"

    def _last_line(cmd: str) -> str:
        lines = backend.run(cmd, check=False).stdout.strip().splitlines()
        return lines[-1] if lines else ""

    with _timed("path resolution", progress):
        old_src = _last_line(f"realpath {workspace}/pytorch")
        new_src = _last_line(f"realpath {worktree_path}")
    if not old_src or not new_src or old_src == new_src:
        log.info("fast-path skipped: path resolution failed or same path")
        return False

    with _timed("base torch check", progress):
        has_torch = backend.run(
            f"cd /tmp && {base_venv}/bin/python -c 'import torch' 2>/dev/null",
            check=False,
        ).returncode
    if has_torch != 0:
        log.info("fast-path skipped: base venv has no torch")
        return False

    log.info("fast-path clone: %s -> %s", old_src, new_src)
    progress("Cloning base venv (fast path)...")
    with _timed("venv clone", progress):
        for cp_flags in ("-al", "-a"):
            if (
                backend.run(
                    f"cp {cp_flags} {base_venv} {job_dir}/.venv", check=False
                ).returncode
                == 0
            ):
                break
            backend.run(f"rm -rf {job_dir}/.venv", check=False)
        else:
            return False

    def _bail() -> bool:
        backend.run(f"rm -rf {job_dir}/.venv", check=False)
        backend.run(f"cd {new_src} && git clean -fdx 2>/dev/null", check=False)
        return False

    progress("Syncing build artifacts into worktree...")
    with _timed("artifact sync", progress):
        r = backend.run(
            f"rsync -a --ignore-existing --exclude='.git' --exclude='build' "
            f"--exclude='__pycache__' --link-dest={old_src} {old_src}/ {new_src}/",
            check=False,
        )
        if r.returncode not in (0, 23):
            progress(f"fast-path bail: rsync failed (rc={r.returncode})")
            return _bail()

    # rsync --link-dest hardlinks .so files to the base workspace.  A later
    # rebuild.sh (pip install -e .) may skip overwriting them if setuptools
    # sees matching size/mtime, leaving the job on stale native code.
    # Break hardlinks now so rebuilds always produce independent copies.
    backend.run(
        f"find {new_src}/torch -name '*.so' -links +1 "
        f"-exec cp --remove-destination {{}} {{}}.tmp \\; "
        f"-exec mv -f {{}}.tmp {{}} \\;",
        check=False,
    )

    job_python = f"{job_dir}/.venv/bin/python"
    sp_dir = _last_line(
        f"{job_python} -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))'"
    )
    if not sp_dir:
        progress("fast-path bail: could not resolve site-packages dir")
        return _bail()

    progress("Rewriting venv paths...")
    with _timed("path rewrite", progress):
        job_venv = f"{job_dir}/.venv"
        resolved_venv = _last_line(f"realpath {job_venv}") or job_venv
        backend.run(
            f'sed -i "s|{base_venv}|{resolved_venv}|g" {job_venv}/bin/activate {job_venv}/bin/activate.csh {job_venv}/bin/activate.fish {job_venv}/bin/activate.nu 2>/dev/null',
            check=False,
        )
        backend.run(
            f"""sed -i "s|^VIRTUAL_ENV=.*|VIRTUAL_ENV='{resolved_venv}'|" {job_venv}/bin/activate 2>/dev/null""",
            check=False,
        )
        backend.run(
            f"""sed -i 's|^setenv VIRTUAL_ENV .*|setenv VIRTUAL_ENV "{resolved_venv}"|' {job_venv}/bin/activate.csh 2>/dev/null""",
            check=False,
        )
        backend.run(
            f"""sed -i 's|^set -gx VIRTUAL_ENV .*|set -gx VIRTUAL_ENV "{resolved_venv}"|' {job_venv}/bin/activate.fish 2>/dev/null""",
            check=False,
        )
        backend.run(
            f'sed -i "1s|#!{base_venv}/bin/python[0-9.]*|#!{resolved_venv}/bin/python|" {job_venv}/bin/* 2>/dev/null',
            check=False,
        )

        backend.run(
            f"for f in {sp_dir}/__editable__*torch* {sp_dir}/torch*.dist-info/direct_url.json; do "
            f'[ -f "$f" ] && sed -i "s|{old_src}|{new_src}|g" "$f"; done',
            check=False,
        )
        backend.run(f"rm -f {sp_dir}/__pycache__/__editable__*torch*.pyc", check=False)

    progress("Installing dev deps (build + test)...")
    with _timed("dev deps", progress):
        r = backend.run(
            f"cd {worktree_path} && "
            f"uv pip install --python {job_python} -r requirements.txt pytest",
            check=False,
            stream=verbose,
        )
        if r.returncode != 0:
            progress(f"fast-path bail: dev deps install failed (rc={r.returncode})")
            return _bail()

    progress("Verifying torch import...")
    with _timed("smoke test", progress):
        smoke = backend.run(
            f"cd /tmp && {job_python} -c "
            f"'import torch; print(torch.__file__, torch.__version__, torch.cuda.is_available())'",
            check=False,
        )
    if smoke.returncode != 0 or new_src not in smoke.stdout:
        reason = (
            f"rc={smoke.returncode} got={smoke.stdout.strip()!r} "
            f"stderr={smoke.stderr.strip()!r} expected={new_src}"
        )
        progress(f"fast-path bail: smoke test failed: {reason}")
        progress(f"Clone verification failed: {reason}")
        progress("Falling back to full install.")
        return _bail()

    log.info("fast-path complete: %s", smoke.stdout.strip())
    progress(f"Editable install complete (cloned) — {smoke.stdout.strip()}")
    return True


def _setup_job_venv(
    backend: Backend,
    job_dir: str,
    worktree_path: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
    build_env_prefix: str = "USE_NINJA=1 ",
) -> None:
    if not _try_clone_base_venv(
        backend, job_dir, worktree_path, verbose=verbose, progress=progress
    ):
        log.info("slow-path: full editable install for %s", job_dir)
        with _timed("venv creation", progress):
            backend.run(f"cd {job_dir} && uv venv --python 3.12")

        job_python = f"{job_dir}/.venv/bin/python"
        progress("Installing dev deps (build + test + Triton)...")
        with _timed("dev deps", progress):
            result = backend.run(
                f"cd {worktree_path} && "
                f"uv pip install --python {job_python} -r requirements.txt pytest",
                check=False,
                stream=verbose,
            )
            result = _chain_result(
                result,
                lambda: backend.run(
                    f"cd {worktree_path} && PATH={job_dir}/.venv/bin:$PATH make triton",
                    check=False,
                    stream=verbose,
                ),
            )
        if result.returncode != 0:
            progress("Dev dependency setup incomplete — agent can install manually.")
        pip_verbose = " -v" if verbose else ""
        pip_cmd = f"uv pip install --python {job_python} --no-build-isolation{pip_verbose} -e ."
        re_cc_cfg = f"{backend.workspace}/.re-cc-config"
        re_cc_check = backend.run(f"cat {re_cc_cfg}", check=False)
        if re_cc_check.returncode == 0 and re_cc_check.stdout.strip().isdigit():
            re_cc_jobs = re_cc_check.stdout.strip()
            pip_cmd = f"re-cc -- {pip_cmd}"
            build_env_prefix = f"MAX_JOBS={re_cc_jobs} {build_env_prefix}"
            progress(f"Using re-cc with MAX_JOBS={re_cc_jobs}")
        progress("Editable install (pytorch)... this takes a few minutes")
        with _timed("editable install", progress):
            result = backend.run(
                f"cd {worktree_path} && {build_env_prefix}{pip_cmd}",
                check=False,
                stream=verbose,
            )
        if result.returncode != 0:
            progress("Editable install failed — agent will need to build manually.")
        else:
            progress("Editable install complete.")


def provision_worktree(
    backend: Backend,
    job_id: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
) -> bool:
    """Create a git worktree and per-worktree venv if they don't already exist.

    Returns True if a new worktree was created, False if reusing existing.
    """
    cb = progress or _noop_progress
    workspace = backend.workspace
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/pytorch"

    backend.run(f"mkdir -p {job_dir}")

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    if worktree_exists.returncode == 0:
        cb("Reusing existing worktree.")
        return False

    cb("Creating worktree with submodules...")
    with _timed("worktree creation", cb):
        backend.run(
            f"cd {workspace}/pytorch && {workspace}/.venv/bin/python tools/create_worktree.py create pytorch "
            f"--parent-dir {job_dir} --commit HEAD",
            stream=verbose,
        )
    cb("Creating per-job venv...")
    from ptq.config import load_config

    _setup_job_venv(
        backend,
        job_dir,
        worktree_path,
        verbose=verbose,
        progress=cb,
        build_env_prefix=load_config().build_env_prefix(),
    )
    return True
