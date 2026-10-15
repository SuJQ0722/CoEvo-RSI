"""Linux Landlock read isolation for MetaAgent sessions (in addition to CLI sandbox).

The repository, journals, sibling workspaces and datasets are absent from the
allowlist. A process cannot escape this boundary with an absolute path or symlink.
Run as a separate launcher, avoiding preexec_fn in the threaded Editor scheduler.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path


def isolated_command(command, workspace, runtime_home):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        str(workspace),
        str(runtime_home),
        *command,
    ]


def restrict(workspace: Path, runtime_home: Path, executable: Path):
    if sys.platform != "linux":
        raise RuntimeError("LINEAGE MetaAgent read isolation requires Linux Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    create, add, enter = 444, 445, 446  # Linux generic syscall numbers (x86_64/aarch64)
    abi = libc.syscall(create, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("LINEAGE requires Landlock ABI >= 1; refusing an unisolated MetaAgent")
    read = (1 << 0) | (1 << 2) | (1 << 3)  # execute, read_file, read_dir
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    rules = Ruleset(handled)
    rules_fd = libc.syscall(create, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if rules_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")

    def allow(path, access):
        path = Path(path)
        if not path.exists():
            return
        if not path.is_dir():
            access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            entry = Beneath(access, fd)
            if libc.syscall(add, rules_fd, 1, ctypes.byref(entry), 0) < 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule: {path}")
        finally:
            os.close(fd)

    try:
        for path in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
            allow(path, read)
        # Devices needed by runtimes; no /proc/*/root or /proc/*/fd escape paths.
        for path in ("/dev/null", "/dev/urandom", "/dev/random", "/dev/zero"):
            allow(path, read | (1 << 1))
        for path in (
            "/proc/cpuinfo",
            "/proc/meminfo",
            "/proc/version",
            "/proc/stat",
            "/proc/sys/kernel/osrelease",
            "/sys/devices/system/cpu",
        ):
            allow(path, read)
        executable = executable.resolve()
        allow(executable.parent, read)
        # Python installations outside /usr need their own standard libraries.
        for prefix in {sys.prefix, sys.base_prefix}:
            allow(Path(prefix) / "lib", read)
            allow(Path(prefix) / "bin", read)
        allow(workspace, handled)
        allow(runtime_home, handled)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(enter, rules_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(rules_fd)


def main():
    workspace, runtime_home = (Path(value).resolve() for value in sys.argv[1:3])
    command = sys.argv[3:]
    executable = Path(shutil.which(command[0]) or command[0])
    runtime_home.mkdir(parents=True, exist_ok=True)
    task_tmp = runtime_home / "tmp"
    task_tmp.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(task_tmp)
    os.environ.pop("PYTHONPATH", None)
    restrict(workspace, runtime_home, executable)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
