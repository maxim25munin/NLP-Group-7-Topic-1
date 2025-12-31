"""Utility for installing the XLM-R optional dependencies from a Jupyter notebook.

Run this script from a notebook cell (``%run scripts/jupyter_xlmr_dependency_setup.py``)
when you see warnings about missing ``transformers`` or ``huggingface_hub``.
It installs the versions specified in ``docs/requirements-transformers.txt`` and
prints the resulting package versions so you can confirm the setup worked.
"""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable

REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "docs" / "requirements-transformers.txt"


def _installed_version(package: str) -> str:
    """Return the installed version of *package* or ``"not installed"``."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def show_dependency_status(packages: Iterable[str]) -> None:
    """Print a table of installed versions for the given *packages*."""

    print("Current dependency versions:")
    for name in packages:
        print(f"- {name}: {_installed_version(name)}")
    print()


def install_xlmr_dependencies() -> None:
    """Install the XLM-R optional dependencies from the pinned requirements file."""

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Expected requirements file at {REQUIREMENTS_FILE!s}; "
            "verify you are running inside the repository root."
        )

    print(f"Installing dependencies from {REQUIREMENTS_FILE}…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    print("Installation complete.\n")


def main() -> None:
    target_packages = [
        "torch",
        "transformers",
        "huggingface_hub",
        "datasets",
        "accelerate",
    ]

    show_dependency_status(target_packages)
    install_xlmr_dependencies()
    show_dependency_status(target_packages)
    print(
        "If you launched this script from a Jupyter notebook, consider restarting the kernel "
        "so the freshly installed packages are available to subsequent cells."
    )


if __name__ == "__main__":
    main()
