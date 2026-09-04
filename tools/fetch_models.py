#!/usr/bin/env python3
"""Download model weights from their original sources.

Weights are never committed to this repository — partly because they are large,
but mainly because redistribution rights differ per model and some of them we
do not have. Every model is fetched from the distributor who actually holds the
rights, at setup time. See PLAN.md §4.

Usage:
    python tools/fetch_models.py            # fetch everything ready to fetch
    python tools/fetch_models.py fire       # fetch one hazard
    python tools/fetch_models.py --list     # show what is available
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "REGISTRY.yaml"


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(url: str, destination: Path) -> Path:
    """Stream a URL to disk with a progress line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "vigia/0.1"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned https URLs
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with destination.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100.0 * written / total
                    print(f"\r  {human(written)} / {human(total)} ({pct:.0f}%)",
                          end="", flush=True)
                else:
                    print(f"\r  {human(written)}", end="", flush=True)
    print()
    return destination


def extract_member(archive: Path, member_name: str, destination: Path) -> Path:
    """Pull a single named file out of a .tar.gz archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        candidates = [
            m for m in tar.getmembers()
            if m.isfile() and Path(m.name).name == member_name
        ]
        if not candidates:
            available = [m.name for m in tar.getmembers() if m.isfile()]
            raise FileNotFoundError(
                f"{member_name!r} not in {archive.name}. Contains: {available}"
            )
        member = candidates[0]
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read {member.name} from {archive.name}")
        with destination.open("wb") as out:
            shutil.copyfileobj(source, out)
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(key: str, spec: dict, force: bool = False) -> bool:
    """Fetch one registry entry. Returns True if the model is on disk after."""
    target = REPO_ROOT / spec["file"]

    if target.exists() and not force:
        print(f"[ok]   {key}\n       already present at {spec['file']} "
              f"({human(target.stat().st_size)})")
        return True

    download_spec = spec.get("download")
    if not download_spec:
        status = spec.get("status", "unknown")
        print(f"[skip] {key}\n       no download URL yet (status: {status})")
        return False

    print(f"[get]  {key}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        url = download_spec["url"]
        archive_member = download_spec.get("archive_member")

        downloaded = download(url, tmp_path / Path(url).name)

        if archive_member:
            extract_member(downloaded, archive_member, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(target))

    print(f"       -> {spec['file']} ({human(target.stat().st_size)})")
    print(f"       sha256 {sha256(target)[:16]}...")
    print(f"       licence: {spec.get('licence', 'UNKNOWN')}  "
          f"author: {spec.get('authors', 'UNKNOWN')}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hazard", nargs="?", default=None,
                        help="only fetch models for this hazard (fire, flood, traffic, drowning)")
    parser.add_argument("--list", action="store_true", help="list registry entries and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    registry = load_registry()
    models = registry.get("models", {})

    if args.list:
        print(f"{'key':<40} {'hazard':<10} {'licence':<14} status")
        print("-" * 84)
        for key, spec in models.items():
            on_disk = (REPO_ROOT / spec["file"]).exists()
            status = "downloaded" if on_disk else spec.get("status", "available")
            print(f"{key:<40} {spec.get('hazard',''):<10} "
                  f"{str(spec.get('licence',''))[:13]:<14} {status}")
        return 0

    selected = {
        k: v for k, v in models.items()
        if args.hazard is None or v.get("hazard") == args.hazard
    }
    if not selected:
        print(f"No models registered for hazard {args.hazard!r}", file=sys.stderr)
        return 1

    fetched = sum(fetch_model(k, v, force=args.force) for k, v in selected.items())
    print(f"\n{fetched}/{len(selected)} model(s) ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
