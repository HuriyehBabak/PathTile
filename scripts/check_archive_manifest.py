#!/usr/bin/env python3
"""Check the size and SHA-256 digest of every file in archive_manifest.csv."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    root = parse_args().project_root.expanduser().resolve()
    manifest_path = root / "artifact/archive_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    required = {"path", "size_bytes", "sha256"}
    if set(manifest.columns) != required:
        raise ValueError(f"Unexpected manifest columns: {list(manifest.columns)}")

    for row in manifest.itertuples(index=False):
        relative = Path(str(row.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row.size_bytes):
            raise AssertionError(f"Size mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(row.sha256):
            raise AssertionError(f"SHA-256 mismatch: {relative}")

    print(f"Archive integrity: PASS ({len(manifest)} files)")


if __name__ == "__main__":
    main()
