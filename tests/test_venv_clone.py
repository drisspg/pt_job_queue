from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock

from ptq.application.run_service import _try_clone_base_venv


def _cp(returncode: int = 0, stdout: str = "") -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=returncode, stdout=stdout, stderr="")


WORKSPACE = "/home/user/ws"
JOB_DIR = f"{WORKSPACE}/jobs/j-123"
WORKTREE = f"{JOB_DIR}/pytorch"
BASE_VENV = f"{WORKSPACE}/.venv"
JOB_VENV = f"{JOB_DIR}/.venv"
OLD_SRC = f"{WORKSPACE}/pytorch"
NEW_SRC = WORKTREE
SP_DIR = f"{JOB_VENV}/lib/python3.12/site-packages"
SMOKE_STDOUT = f"{NEW_SRC}/torch/__init__.py 2.7.0 True"


def _make_backend(*, torch_import_ok: bool = True) -> MagicMock:
    backend = MagicMock()
    backend.workspace = WORKSPACE

    def run_side(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
        if f"realpath {WORKSPACE}/pytorch" in cmd:
            return _cp(stdout=f"{OLD_SRC}\n")
        if f"realpath {WORKTREE}" in cmd:
            return _cp(stdout=f"{NEW_SRC}\n")
        if "import torch; print" in cmd:
            return _cp(stdout=SMOKE_STDOUT)
        if "import torch" in cmd:
            return _cp(returncode=0 if torch_import_ok else 1)
        if cmd.startswith("cp "):
            return _cp(0)
        if "rsync" in cmd:
            return _cp(0)
        if "sysconfig" in cmd:
            return _cp(stdout=f"{SP_DIR}\n")
        if "uv pip install" in cmd:
            return _cp(0)
        return _cp(0)

    backend.run = MagicMock(side_effect=run_side)
    return backend


def _all_cmds(backend: MagicMock) -> list[str]:
    return [
        call.args[0]
        for call in backend.run.call_args_list
        if call.args and isinstance(call.args[0], str)
    ]


class TestActivateRewrite:
    def test_rewrites_activate_scripts(self):
        backend = _make_backend()
        _try_clone_base_venv(backend, JOB_DIR, WORKTREE)
        cmds = _all_cmds(backend)

        activate_seds = [c for c in cmds if "activate" in c and "sed" in c]
        assert len(activate_seds) == 4

        generic = activate_seds[0]
        assert f"s|{BASE_VENV}|{JOB_VENV}|g" in generic
        for script in ("activate", "activate.csh", "activate.fish", "activate.nu"):
            assert f"{JOB_VENV}/bin/{script}" in generic

        bash_fix = activate_seds[1]
        assert "^VIRTUAL_ENV=" in bash_fix
        assert f"VIRTUAL_ENV='{JOB_VENV}'" in bash_fix

        csh_fix = activate_seds[2]
        assert "^setenv VIRTUAL_ENV" in csh_fix
        assert f'VIRTUAL_ENV "{JOB_VENV}"' in csh_fix

        fish_fix = activate_seds[3]
        assert "^set -gx VIRTUAL_ENV" in fish_fix
        assert f'VIRTUAL_ENV "{JOB_VENV}"' in fish_fix


class TestShebangRewrite:
    def test_rewrites_shebangs_in_bin(self):
        backend = _make_backend()
        _try_clone_base_venv(backend, JOB_DIR, WORKTREE)
        cmds = _all_cmds(backend)

        shebang_seds = [c for c in cmds if "1s|#!" in c and "bin/python" in c]
        assert len(shebang_seds) == 1
        cmd = shebang_seds[0]
        assert f"#!{BASE_VENV}/bin/python[0-9.]*" in cmd
        assert f"#!{JOB_VENV}/bin/python" in cmd
        assert f"{JOB_VENV}/bin/*" in cmd


class TestEditableInstallRewrite:
    def test_rewrites_editable_paths(self):
        backend = _make_backend()
        _try_clone_base_venv(backend, JOB_DIR, WORKTREE)
        cmds = _all_cmds(backend)

        editable_seds = [c for c in cmds if "__editable__" in c and "sed" in c]
        assert len(editable_seds) == 1
        cmd = editable_seds[0]
        assert f"s|{OLD_SRC}|{NEW_SRC}|g" in cmd
        assert "direct_url.json" in cmd

    def test_clears_editable_pyc_cache(self):
        backend = _make_backend()
        _try_clone_base_venv(backend, JOB_DIR, WORKTREE)
        cmds = _all_cmds(backend)

        pyc_cmds = [c for c in cmds if "__editable__" in c and "rm -f" in c]
        assert len(pyc_cmds) == 1
        assert ".pyc" in pyc_cmds[0]


class TestHardlinkBreaking:
    def test_breaks_so_hardlinks_after_rsync(self):
        backend = _make_backend()
        _try_clone_base_venv(backend, JOB_DIR, WORKTREE)
        cmds = _all_cmds(backend)

        find_cmds = [c for c in cmds if "links +1" in c and ".so" in c]
        assert len(find_cmds) == 1
        cmd = find_cmds[0]
        assert f"{NEW_SRC}/torch" in cmd
        assert "--remove-destination" in cmd

        rsync_idx = next(i for i, c in enumerate(cmds) if "rsync" in c)
        find_idx = next(i for i, c in enumerate(cmds) if "links +1" in c)
        assert find_idx > rsync_idx


class TestFastPathSkips:
    def test_skips_when_realpath_fails(self):
        backend = MagicMock()
        backend.workspace = WORKSPACE
        backend.run = MagicMock(return_value=_cp(stdout=""))

        assert _try_clone_base_venv(backend, JOB_DIR, WORKTREE) is False

    def test_skips_when_no_torch_in_base(self):
        backend = _make_backend(torch_import_ok=False)
        assert _try_clone_base_venv(backend, JOB_DIR, WORKTREE) is False

        cmds = _all_cmds(backend)
        assert not any("sed" in c for c in cmds)

    def test_skips_when_same_path(self):
        backend = MagicMock()
        backend.workspace = WORKSPACE
        backend.run = MagicMock(return_value=_cp(stdout=f"{OLD_SRC}\n"))

        assert _try_clone_base_venv(backend, JOB_DIR, OLD_SRC) is False


class TestCloneSuccess:
    def test_returns_true_on_success(self):
        backend = _make_backend()
        assert _try_clone_base_venv(backend, JOB_DIR, WORKTREE) is True

    def test_bails_when_smoke_test_fails(self):
        backend = MagicMock()
        backend.workspace = WORKSPACE

        def run_side(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
            if f"realpath {WORKSPACE}/pytorch" in cmd:
                return _cp(stdout=f"{OLD_SRC}\n")
            if f"realpath {WORKTREE}" in cmd:
                return _cp(stdout=f"{NEW_SRC}\n")
            if "import torch" in cmd and "print" not in cmd and "sysconfig" not in cmd:
                return _cp(0)
            if "cp " in cmd:
                return _cp(0)
            if "rsync" in cmd:
                return _cp(0)
            if "sysconfig" in cmd:
                return _cp(stdout=f"{SP_DIR}\n")
            if "uv pip install" in cmd:
                return _cp(0)
            if "import torch; print" in cmd:
                return _cp(returncode=1, stdout="ImportError")
            return _cp(0)

        backend.run = MagicMock(side_effect=run_side)
        assert _try_clone_base_venv(backend, JOB_DIR, WORKTREE) is False
