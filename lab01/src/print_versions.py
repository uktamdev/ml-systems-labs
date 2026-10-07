"""Prints the Python version and the version of every package in requirements.txt."""
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

# requirements.txt is in the repo root, two folders above this file
REQ_FILE = Path(__file__).resolve().parents[2] / "requirements.txt"


def package_names():
    names = []
    for line in REQ_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "numpy==1.26.4" -> "numpy"
        names.append(re.split(r"[=<>!~ ]", line, maxsplit=1)[0])
    return names


def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("-" * 40)
    for name in package_names():
        try:
            print(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")


if __name__ == "__main__":
    main()