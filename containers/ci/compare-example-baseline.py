#!/usr/bin/env python3
"""Compare or update small text baselines for example regression runs."""

from __future__ import annotations

import argparse
import difflib
import filecmp
import re
import shutil
import sys
from pathlib import Path


TEXT_EXTENSIONS = {".csv", ".json", ".txt"}
VOLATILE_PATTERNS = [
    re.compile(r"Simulation finished after .*$"),
    re.compile(r"Program finished after .*$"),
]


def normalize_text(path: Path) -> str:
    text = path.read_text(errors="replace")
    lines = []
    for line in text.splitlines():
        for pattern in VOLATILE_PATTERNS:
            line = pattern.sub("<elapsed-time>", line)
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def collect_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            files.append(path.relative_to(root))
    return sorted(files)


def write_normalized_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in collect_files(source):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(normalize_text(source / rel))


def compare_trees(actual: Path, baseline: Path) -> tuple[bool, str]:
    actual_files = collect_files(actual)
    baseline_files = collect_files(baseline)
    all_files = sorted(set(actual_files) | set(baseline_files))
    chunks: list[str] = []
    ok = True

    for rel in all_files:
        actual_path = actual / rel
        baseline_path = baseline / rel
        if rel not in actual_files:
            ok = False
            chunks.append(f"Missing actual file: {rel}\n")
            continue
        if rel not in baseline_files:
            ok = False
            chunks.append(f"Unexpected actual file: {rel}\n")
            continue

        if filecmp.cmp(actual_path, baseline_path, shallow=False):
            continue
        ok = False
        actual_lines = actual_path.read_text(errors="replace").splitlines(keepends=True)
        baseline_lines = baseline_path.read_text(errors="replace").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                baseline_lines,
                actual_lines,
                fromfile=f"baseline/{rel}",
                tofile=f"actual/{rel}",
            )
        )

    return ok, "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--diff-output", type=Path)
    args = parser.parse_args()

    actual_norm = args.work / "actual-normalized"
    write_normalized_tree(args.actual, actual_norm)
    if not collect_files(actual_norm):
        print(f"No comparable text outputs found under: {args.actual}", file=sys.stderr)
        return 2

    if args.update:
        write_normalized_tree(args.actual, args.baseline)
        return 0

    if not args.baseline.exists():
        print(f"Baseline does not exist: {args.baseline}", file=sys.stderr)
        return 2

    ok, diff = compare_trees(actual_norm, args.baseline)
    if args.diff_output:
        args.diff_output.parent.mkdir(parents=True, exist_ok=True)
        args.diff_output.write_text(diff)
    if not ok:
        if diff:
            sys.stderr.write(diff)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
