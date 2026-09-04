#!/usr/bin/env python3
"""Package the paper for arXiv.

    python docs/make_arxiv.py        # or: make arxiv

Produces dist/vigia-arxiv.tar.gz containing exactly what arXiv's compiler
needs and nothing it doesn't:

  vigia.tex, preamble.tex, sections/, generated/   — the LaTeX sources
  vigia.bbl                                        — the resolved bibliography
                                                     (arXiv does not run
                                                     bibtex; omitting the
                                                     .bbl yields [?] marks)
  figures/*.pdf                                    — only the figures the
                                                     paper actually includes
  refs.bib                                         — harmless, aids readers

The build runs FIRST, so the bundle can never contain sources that disagree
with eval/results — the same rule as everywhere else in this project.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / "docs" / "paper"
FIGURES = REPO_ROOT / "docs" / "figures"
DIST = REPO_ROOT / "dist"


def main() -> int:
    # 1. Rebuild from the measurements, keeping the .bbl.
    print("rebuilding the paper (figures -> numbers -> LaTeX)")
    build = subprocess.run(
        [sys.executable, str(REPO_ROOT / "docs" / "build_paper.py")],
        capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stdout[-2000:], build.stderr[-2000:], file=sys.stderr)
        return 1

    print("recompiling with --keep-intermediates for the .bbl")
    keep = subprocess.run(
        ["tectonic", "-X", "compile", str(PAPER / "vigia.tex"),
         "--outdir", str(PAPER), "--keep-intermediates"],
        capture_output=True, text=True)
    if keep.returncode != 0 or not (PAPER / "vigia.bbl").exists():
        print("no .bbl produced — arXiv would render [?] citations",
              file=sys.stderr)
        print(keep.stdout[-2000:], keep.stderr[-1500:], file=sys.stderr)
        return 1

    # 2. Stage the bundle.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "vigia-arxiv"
        (stage / "sections").mkdir(parents=True)
        (stage / "generated").mkdir()
        (stage / "figures").mkdir()

        for name in ("vigia.tex", "preamble.tex", "refs.bib", "vigia.bbl"):
            shutil.copy2(PAPER / name, stage / name)
        for f in (PAPER / "sections").glob("*.tex"):
            shutil.copy2(f, stage / "sections" / f.name)
        for f in (PAPER / "generated").glob("*.tex"):
            shutil.copy2(f, stage / "generated" / f.name)

        # Only the figures the sources reference, as vector PDF.
        wanted = set()
        for f in [stage / "vigia.tex", *(stage / "sections").glob("*.tex")]:
            wanted.update(re.findall(
                r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                f.read_text(encoding="utf-8")))
        for name in sorted(wanted):
            source = FIGURES / Path(name).name
            if source.suffix == "":
                source = source.with_suffix(".pdf")
            if not source.exists():
                print(f"referenced figure missing: {source}", file=sys.stderr)
                return 1
            shutil.copy2(source, stage / "figures" / source.name)

        DIST.mkdir(exist_ok=True)
        out = DIST / "vigia-arxiv.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    tar.add(f, arcname=str(f.relative_to(stage)))

        names = [str(f.relative_to(stage)) for f in sorted(stage.rglob("*"))
                 if f.is_file()]
        print(f"\nwrote {out.relative_to(REPO_ROOT)} "
              f"({out.stat().st_size / 1024:.0f} KB, {len(names)} files):")
        for name in names:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
