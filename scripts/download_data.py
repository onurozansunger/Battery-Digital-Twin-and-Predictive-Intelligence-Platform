#!/usr/bin/env python3
"""Fetch and unpack the NASA Ames PCoE battery aging dataset.

    python scripts/download_data.py
    python scripts/download_data.py --dest data/raw/nasa --force

The archive is ~200 MB and contains six nested zips; this script flattens them
into ``<dest>/mat/*.mat``, which is the layout ``NASABatterySource`` expects.

Source
------
NASA Ames Prognostics Data Repository, "Battery Data Set" (Saha & Goebel, 2007),
mirrored on the public ``phm-datasets`` S3 bucket. The data is US-government work
and is redistributed by NASA for research use; please cite the original authors.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
EXPECTED_MB = 200
CHUNK = 1 << 20


def _download(url: str, dest: Path) -> Path:
    print(f"Downloading {url}\n  -> {dest}")
    with urlopen(url) as response:  # noqa: S310 - fixed, hard-coded NASA mirror
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f} %)", end=""
                    )
        print()
    return dest


def _extract(archive: Path, mat_dir: Path) -> int:
    """Unpack the outer archive, then every nested zip, flattening the .mat files."""
    mat_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(stage)

        for nested in sorted(stage.rglob("*.zip")):
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(stage / "inner")

        count = 0
        for mat in sorted(stage.rglob("*.mat")):
            target = mat_dir / mat.name
            if not target.exists():
                shutil.move(str(mat), target)
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", default="data/raw/nasa", help="Destination directory.")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if .mat files exist."
    )
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded zip.")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    mat_dir = dest / "mat"
    existing = sorted(mat_dir.glob("*.mat")) if mat_dir.is_dir() else []
    if existing and not args.force:
        print(f"{len(existing)} .mat files already present in {mat_dir}. Use --force to refetch.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "nasa_battery.zip"

    if not archive.is_file() or args.force:
        try:
            _download(args.url, archive)
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}", file=sys.stderr)
            print(
                "\nManual alternative: download the 'Battery Data Set' from\n"
                "  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
                f"and unzip every nested archive into {mat_dir}",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"Using cached archive {archive}")

    n = _extract(archive, mat_dir)
    print(f"Extracted {n} .mat files to {mat_dir}")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed {archive} (pass --keep-archive to retain it)")

    total = len(sorted(mat_dir.glob("*.mat")))
    print(f"\nDone. {total} battery files available.")
    print("Next:  python scripts/run_pipeline.py --config configs/default.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
