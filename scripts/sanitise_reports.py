#!/usr/bin/env python3
"""Strip absolute developer paths from committed report artifacts.

Every report embeds the resolved configuration, and `PathsConfig` absolutises
every directory against the project root — so a committed `metrics.json` carried
the author's home directory into the repository. That is a small privacy leak, it
makes diffs between machines noisy, and it invites a reader to think the paths
mean something.

This rewrites any occurrence of the project root with the literal `<project_root>`
in the artifacts that get committed, leaving the values structurally intact.
Idempotent: running it twice changes nothing.

    python scripts/sanitise_reports.py
    python scripts/sanitise_reports.py --check      # exit 1 if anything is left
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_rul.config import load_config, project_root  # noqa: E402
from battery_rul.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

PLACEHOLDER = "<project_root>"

#: Extensions worth rewriting. Parquet and PNG are binary and are not committed
#: with embedded paths.
TEXT_SUFFIXES = {".json", ".md", ".csv", ".yaml", ".yml"}


def _targets(cfg) -> list[Path]:
    roots = [cfg.paths.reports_dir, cfg.paths.processed_dir, cfg.artifacts.root]
    files: list[Path] = []
    for root in roots:
        if not Path(root).is_dir():
            continue
        files.extend(
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report offending files and exit non-zero instead of rewriting.",
    )
    args = parser.parse_args(argv)

    path = Path(args.config)
    cfg = load_config(path if path.is_file() else None)
    setup_logging(force=True)

    root = str(project_root())
    home = str(Path.home())
    offenders: list[Path] = []
    rewritten = 0

    for file in _targets(cfg):
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if root not in text and home not in text:
            continue
        offenders.append(file)
        if args.check:
            continue
        # Longest first, so the project root is replaced before the home prefix
        # it sits inside.
        cleaned = text.replace(root, PLACEHOLDER)
        cleaned = cleaned.replace(home, "<home>")
        if cleaned != text:
            file.write_text(cleaned, encoding="utf-8")
            rewritten += 1

    if args.check:
        if offenders:
            logger.error(
                "%d artifact(s) still contain absolute machine paths: %s",
                len(offenders),
                [str(p.name) for p in offenders[:20]],
            )
            return 1
        logger.info("No absolute machine paths in committed artifacts.")
        return 0

    logger.info("Sanitised %d/%d artifact(s).", rewritten, len(_targets(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
