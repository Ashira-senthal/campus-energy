"""
Download the three BDG2 files this project needs into ``data/raw/``.

    .venv/bin/python src/download_data.py          # skip files already present
    .venv/bin/python src/download_data.py --force  # re-download

Why this script exists: the raw CSVs total ~186 MB and are git-ignored, so a fresh
clone has to fetch them. The repository stores them in **Git LFS**, which means
the obvious URL (``raw.githubusercontent.com/...``) returns a ~130-byte pointer
file instead of data. These are the LFS media URLs that serve the real bytes, and
every download is size-checked so a pointer file can never be mistaken for data.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "buds-lab/building-data-genome-project-2/master"
)

# local name -> (path inside the BDG2 repo, expected size in bytes)
FILES: dict[str, tuple[str, int]] = {
    "electricity_cleaned.csv": ("data/meters/cleaned/electricity_cleaned.csv", 174_977_911),
    "weather.csv": ("data/weather/weather.csv", 19_457_782),
    "metadata.csv": ("data/metadata/metadata.csv", 272_024),
}


def _human(size: int) -> str:
    return f"{size / 1_048_576:.1f} MB"


def download(force: bool = False) -> int:
    """Fetch every missing file. Returns a process exit code."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for local_name, (remote_path, expected_size) in FILES.items():
        target = RAW_DIR / local_name
        if target.exists() and not force:
            size = target.stat().st_size
            if size == expected_size:
                print(f"[skip] {local_name} already present ({_human(size)})")
                continue
            print(f"[warn] {local_name} has the wrong size ({size} bytes); re-downloading")
        if target.exists() and force:
            target.unlink()

        url = f"{BASE_URL}/{remote_path}"
        print(f"[get ] {local_name} <- {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as error:  # network, DNS, 404, ...
            print(f"[fail] {local_name}: {error}")
            failures += 1
            continue

        size = target.stat().st_size
        if size != expected_size:
            print(
                f"[fail] {local_name}: got {size} bytes, expected {expected_size}. "
                "That usually means an LFS pointer file was served instead of the data."
            )
            failures += 1
        else:
            print(f"[ok  ] {local_name} ({_human(size)})")

    if failures:
        print(
            "\nSome files could not be downloaded. Download them manually from\n"
            "  https://github.com/buds-lab/building-data-genome-project-2\n"
            f"and place them in {RAW_DIR} using the names above.",
            file=sys.stderr,
        )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    return download(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
