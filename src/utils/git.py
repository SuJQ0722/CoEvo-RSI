"""Read explicitly selected checkouts across shared-volume/container users."""
from pathlib import Path
import subprocess


def read_checkout(path: Path, *arguments: str) -> str:
    """Trust only this checkout for this Git process; leave user config untouched."""
    checkout = str(path.resolve())
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", checkout, *arguments],
        text=True,
    ).strip()
