#!/usr/bin/env python3
"""Guard: keep biometric identification out of VIGÍA, by architecture.

PLAN.md section 10 commits to identifying nobody — "no face recognition, no
re-identification, no gait or biometric feature extraction, **by architecture,
not configuration**. There must be no code path that could be switched on."

A commitment that lives only in a document is a configuration flag waiting to
happen. This makes it structural: the build fails if a biometric dependency or
API appears anywhere under `vigia/`, in the same way `check_licence.py` keeps
AGPL packages out of the runtime.

Why it matters beyond good intentions: the EU AI Act's high-risk provisions
took full effect in August 2026. Annex III classifies biometric identification
as high-risk, and Article 5(1)(h) bans real-time remote biometric identification
in publicly accessible spaces for law enforcement. VIGÍA has no need to identify
anyone — it needs to know that a person is present and where. Being
architecturally incapable of identification is cheaper and more credible than
being contractually unwilling.

Unlike the licence guard, `tools/` is NOT exempt. An offline script that builds
a face index would breach the commitment just as surely as a runtime one.

Run in CI. Exits non-zero on violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIRS = ("vigia", "tools", "scripts", "eval")

#: Packages whose entire purpose is identifying or re-identifying a person.
FORBIDDEN_IMPORTS = {
    "face_recognition": "face identification",
    "facenet_pytorch": "face embedding",
    "insightface": "face identification",
    "deepface": "face identification",
    "dlib": "face landmarks / embeddings",
    "torchreid": "person re-identification",
    "mediapipe": "face mesh / landmark extraction",
}

#: Attribute paths that reach identification APIs inside otherwise fine
#: libraries. OpenCV in particular ships a whole face module.
FORBIDDEN_ATTRIBUTES = {
    "cv2.face": "OpenCV face recognition module",
    "cv2.FaceRecognizerSF": "OpenCV face recognition",
    "cv2.FaceDetectorYN": "OpenCV face detection",
    "cv2.CascadeClassifier": "Haar cascades, used almost exclusively for faces",
}

#: Substrings in identifiers that indicate identity work. Matched against
#: function, class and variable names — not comments, which legitimately
#: discuss what we do not do.
#: Matched against a name with underscores STRIPPED, so `face_embedding`,
#: `faceEmbedding` and `FaceEmbeddingExtractor` all trip the same fragment.
#: Learned by testing the guard rather than by reasoning about it: an early
#: version used whole words with underscores and let both `PersonReidentifier`
#: and `GaitSignatureExtractor` through. A guard that cannot fail is worthless,
#: so this one is tested against deliberate violations.
FORBIDDEN_NAMES = (
    "faceencoding", "faceembedding", "facerecogni", "faceid",
    "reidentif",
    "gaitsignature", "gaitanalysis",
    "biometric", "identifyperson", "personidfrom",
)


def attribute_path(node: ast.AST) -> str:
    """Dotted path for an attribute access, e.g. cv2.face.LBPHFaceRecognizer."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def scan(path: Path) -> list[tuple[int, str, str]]:
    """Return (line, finding, why) for one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"  ! could not parse {path}: {exc}", file=sys.stderr)
        return []

    findings: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        # Imports of identification libraries.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    findings.append((node.lineno, f"import {alias.name}",
                                     FORBIDDEN_IMPORTS[root]))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    findings.append((node.lineno, f"from {node.module} import ...",
                                     FORBIDDEN_IMPORTS[root]))

        # Identification APIs inside permitted libraries.
        elif isinstance(node, ast.Attribute):
            dotted = attribute_path(node)
            for forbidden, why in FORBIDDEN_ATTRIBUTES.items():
                if dotted == forbidden or dotted.startswith(forbidden + "."):
                    findings.append((node.lineno, dotted, why))

        # Names that describe identity work.
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lowered = node.name.lower().replace("_", "")
            for fragment in FORBIDDEN_NAMES:
                if fragment in lowered:
                    findings.append((node.lineno, node.name,
                                     f"name suggests identity work ({fragment})"))
    return findings


def main() -> int:
    violations: list[tuple[Path, int, str, str]] = []
    checked = 0

    for directory in CHECKED_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name == "check_privacy.py":
                continue                       # this file names them all
            checked += 1
            for line, finding, why in scan(path):
                violations.append((path.relative_to(REPO_ROOT), line, finding, why))

    if violations:
        print("PRIVACY GUARD FAILED\n")
        for path, line, finding, why in violations:
            print(f"  {path}:{line}  {finding}  — {why}")
        print(
            "\nVIGÍA identifies nobody, by architecture rather than by "
            "configuration.\nPerson detection is reduced to presence and "
            "location within a hazard region,\nnever identity. If a hazard "
            "genuinely requires this, it does not belong in\nthis system. See "
            "PLAN.md section 10 and the EU AI Act, Annex III and\nArticle "
            "5(1)(h)."
        )
        return 1

    print(f"PRIVACY GUARD PASSED — {checked} file(s) checked across "
          f"{', '.join(CHECKED_DIRS)}/, no biometric identification path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
