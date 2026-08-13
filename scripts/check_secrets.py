#!/usr/bin/env python
"""Refuse to ship a repository that contains a credential.

A deliberately small, dependency-free scanner over the *tracked* files. It is
not a replacement for a managed secret scanner — it will not catch a
high-entropy string with no recognisable prefix — and it is not trying to be:
its job is to fail loudly on the mistakes that actually happen (a pasted API
key, a private key file, a filled-in .env) at the moment they are about to be
committed, without needing an account anywhere.

    python scripts/check_secrets.py            # scan the tracked files
    python scripts/check_secrets.py --all      # scan the working tree too

Exit codes: 0 clean, 1 findings.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Patterns that are worth failing a build over. Each is a *prefix* or a
#: structural marker, not an entropy heuristic: entropy alone flags every
#: fingerprint and checksum in this repository, and a check that cries wolf is a
#: check people disable.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("AWS access key id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("AWS secret access key", r"aws_secret_access_key\s*=\s*['\"]?[A-Za-z0-9/+=]{40}"),
    ("GitHub token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    ("Slack token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("Google API key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("Anthropic API key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    ("OpenAI API key", r"\bsk-[A-Za-z0-9]{32,}\b"),
    ("Private key block", r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP)? ?PRIVATE KEY-----"),
    ("JSON web token", r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    (
        "Hardcoded password assignment",
        r"(?i)\b(password|passwd|secret|api_key|apikey|access_token)\s*[:=]\s*"
        r"['\"](?!<|\$\{|placeholder|changeme|example|your-|xxx|\.\.\.)[^'\"\s]{8,}['\"]",
    ),
    (
        "Database URL with credentials",
        r"(?i)\b(postgres|postgresql|mysql|mongodb)://[^:\s]+:[^@\s]+@",
    ),
)

#: Files whose whole purpose is to describe these patterns.
ALLOWLIST_FILES = frozenset(
    {
        "scripts/check_secrets.py",
        "docs/SECURITY.md",
        "SECURITY.md",
    }
)

#: Extensions that never contain source-level credentials but do contain long
#: random-looking strings (hashes, parquet, images).
SKIP_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".parquet",
        ".pkl",
        ".joblib",
        ".zip",
        ".mat",
        ".db",
        ".ipynb",
    }
)

MAX_BYTES = 2_000_000


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return sorted(p for p in Path().rglob("*") if p.is_file())
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def working_tree_files() -> list[Path]:
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
    out: list[Path] = []
    for path in Path().rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        out.append(path)
    return sorted(out)


def scan(paths: list[Path]) -> list[tuple[Path, int, str, str]]:
    compiled = [(name, re.compile(pattern)) for name, pattern in PATTERNS]
    findings: list[tuple[Path, int, str, str]] = []

    for path in paths:
        if str(path).replace("\\", "/") in ALLOWLIST_FILES:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for number, line in enumerate(text.splitlines(), start=1):
            if len(line) > 2000:
                continue
            for name, pattern in compiled:
                if pattern.search(line):
                    findings.append((path, number, name, line.strip()[:120]))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all", action="store_true", help="Scan the working tree, not only tracked files."
    )
    args = parser.parse_args(argv)

    paths = working_tree_files() if args.all else tracked_files()
    findings = scan(paths)

    if not findings:
        print(f"No credential patterns found in {len(paths)} file(s).")
        return 0

    print(f"Potential secrets found in {len({f[0] for f in findings})} file(s):\n")
    for path, number, name, excerpt in findings:
        print(f"  {path}:{number}  [{name}]")
        print(f"      {excerpt}")
    print(
        "\nIf a finding is a false positive, make the value obviously a placeholder "
        "(`<your-token>`, `${TOKEN}`) rather than adding an exception here — an "
        "exception list is where a real key eventually hides."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
