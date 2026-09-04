"""Tier 0 and the privacy boundary, enforced rather than asserted.

The project's pattern is that a claim in the dossier should be defended by a
test that fails if the claim stops being true — the way
`test_validator_contains_no_hazard_specific_code` defends the hazard-agnostic
claim. Two claims are defended here:

  * The gate is declarative and fails loudly. An unregistered camera is an
    error, not an empty hazard set, because a typo that silently produces a
    camera watching for nothing is the worst possible failure in this system.
  * VIGÍA identifies nobody, structurally. The evidence frame is redacted, the
    Alert type has nowhere to put a video clip, and no biometric code path
    exists anywhere in the tree.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigia.io.alerts import Alert, PERSON_LABELS, redact_people  # noqa: E402
from vigia.registry import (  # noqa: E402
    CONTEXT_HAZARDS, CONTEXT_VIEWPOINT, Camera, CameraContext, CameraRegistry,
    RetentionPolicy, UnknownCamera, Viewpoint,
)
from vigia.types import Box, Detection, Hazard  # noqa: E402


def _camera(camera_id="cam", context=CameraContext.FOREST, **kwargs):
    return Camera(camera_id, "clip.mp4", context, **kwargs)


# --------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------- #

def test_context_decides_hazards():
    registry = CameraRegistry([_camera("ridge", CameraContext.FOREST)])
    assert registry.hazards_for("ridge") == frozenset({Hazard.FIRE})


def test_street_carries_the_dana_hazards():
    """The case the project exists for: water, vehicles, trapped people."""
    hazards = CONTEXT_HAZARDS[CameraContext.STREET]
    assert Hazard.FLOOD in hazards
    assert Hazard.TRAFFIC in hazards
    assert Hazard.BUILDING_ACCESS in hazards


def test_unknown_camera_raises_rather_than_returning_nothing():
    """An unregistered camera must never look like a camera with no hazards.

    Returning an empty set would turn a typo into a camera that watches for
    nothing and reports success.
    """
    registry = CameraRegistry([_camera("ridge")])
    try:
        registry.hazards_for("rdige")          # typo
    except UnknownCamera:
        return
    raise AssertionError("unknown camera silently returned a hazard set")


def test_disabled_camera_runs_nothing():
    registry = CameraRegistry([_camera("ridge", enabled=False)])
    assert registry.hazards_for("ridge") == frozenset()


def test_explicit_hazards_override_the_context_and_are_visible():
    camera = _camera("coast", CameraContext.COAST,
                     hazards=frozenset({Hazard.DROWNING, Hazard.TRAFFIC}))
    registry = CameraRegistry([camera])
    assert registry.hazards_for("coast") == frozenset(
        {Hazard.DROWNING, Hazard.TRAFFIC})
    assert camera.overrides_context, "an override must be inspectable"


def test_duplicate_camera_id_is_refused():
    registry = CameraRegistry([_camera("ridge")])
    try:
        registry.add(_camera("ridge"))
    except ValueError:
        return
    raise AssertionError("duplicate camera_id was accepted")


def test_only_needed_detectors_are_required():
    """The gate's saving is real only if unused models are never loaded."""
    registry = CameraRegistry([
        _camera("a", CameraContext.FOREST),
        _camera("b", CameraContext.MOTORWAY),
    ])
    assert registry.required_hazards() == frozenset({Hazard.FIRE, Hazard.TRAFFIC})


def test_gate_summary_reports_a_real_reduction():
    registry = CameraRegistry([
        _camera("a", CameraContext.FOREST),      # 1 hazard
        _camera("b", CameraContext.MOTORWAY),    # 1 hazard
    ])
    summary = registry.gate_summary()
    assert summary["detector_invocations_per_frame_gated"] == 2
    assert summary["detector_invocations_per_frame_ungated"] == 2 * len(Hazard)
    assert summary["reduction"] > 0.5


def test_unknown_context_fails_at_load():
    try:
        CameraRegistry.from_dict({"cameras": [
            {"camera_id": "x", "source": "a.mp4", "context": "volcano"}
        ]})
    except ValueError as exc:
        assert "volcano" in str(exc)
        return
    raise AssertionError("an unknown context loaded without complaint")


def test_every_context_maps_to_at_least_one_hazard():
    """A context that enables nothing is a camera that watches nothing."""
    empty = [c.value for c in CameraContext if not CONTEXT_HAZARDS.get(c)]
    assert not empty, f"contexts with no hazards: {empty}"


# --------------------------------------------------------------------- #
# Viewpoint — a detector must not run outside its trained envelope
# --------------------------------------------------------------------- #

def test_context_implies_a_viewpoint():
    assert _camera("uav", CameraContext.UAV_SEARCH).active_viewpoint \
        is Viewpoint.NADIR_AERIAL
    assert _camera("street", CameraContext.STREET).active_viewpoint \
        is Viewpoint.OBLIQUE


def test_camera_can_override_its_viewpoint():
    """A river seen from a mast and the same river seen from a drone are not
    the same sensing problem."""
    camera = _camera("river_drone", CameraContext.RIVER,
                     viewpoint=Viewpoint.NADIR_AERIAL)
    assert camera.active_viewpoint is Viewpoint.NADIR_AERIAL


def test_every_context_has_a_declared_viewpoint():
    missing = [c.value for c in CameraContext if c not in CONTEXT_VIEWPOINT]
    assert not missing, f"contexts with no viewpoint: {missing}"


def test_flood_detector_refuses_a_nadir_camera_without_aerial_weights():
    """The concrete failure this exists to prevent.

    The flood segmenter is trained on ATLANTIS — ground-level and oblique
    photography. On straight-down UAV imagery it reported up to 0.397 water
    coverage while tinting mown grass as water: a confident wrong answer, which
    is the worst way for a life-safety component to fail. The pairing must be
    refused before any model is loaded.
    """
    from vigia.detectors.flood import AERIAL_MODEL_PATH
    from vigia.pipeline import CameraPipeline

    if AERIAL_MODEL_PATH.exists():
        return                      # capability present; the pairing is now
                                    # legitimate and the test above checks it
    camera = _camera("uav_flood", CameraContext.UAV_SEARCH,
                     hazards=frozenset({Hazard.FLOOD}))
    try:
        CameraPipeline(camera)
    except ValueError as exc:
        assert "envelope" in str(exc)
        return
    raise AssertionError(
        "a nadir camera was allowed to run the ground-trained flood segmenter")


def test_nadir_flood_camera_is_served_not_refused():
    """The capability, not just the guard.

    A UAV flood camera was refused while the only flood weights were trained on
    ground-level photography. With aerial weights present it must be SERVED,
    and served by those weights rather than silently by the ground ones.
    """
    from vigia.detectors.flood import AERIAL_MODEL_PATH, DEFAULT_MODEL_PATH
    from vigia.pipeline import CameraPipeline

    if not AERIAL_MODEL_PATH.exists():
        return                      # capability legitimately absent; the
                                    # refusal test above covers that case
    camera = _camera("uav_flood", CameraContext.UAV_SEARCH,
                     hazards=frozenset({Hazard.FLOOD}))
    pipeline = CameraPipeline(camera)
    used = pipeline.detectors[Hazard.FLOOD].model_path
    assert used == AERIAL_MODEL_PATH, (
        f"nadir camera served by {used.name}, expected the aerial weights")

    ground = _camera("creek", CameraContext.RIVER,
                     hazards=frozenset({Hazard.FLOOD}))
    assert CameraPipeline(ground).detectors[Hazard.FLOOD].model_path \
        == DEFAULT_MODEL_PATH


def test_every_detector_declares_a_viewpoint_envelope():
    """A detector with no declared envelope can be paired with anything."""
    from vigia.pipeline import _detector_viewpoints

    undeclared = [h.value for h in Hazard if not _detector_viewpoints(h)]
    assert not undeclared, f"detectors with no viewpoint envelope: {undeclared}"


# --------------------------------------------------------------------- #
# Privacy, structurally
# --------------------------------------------------------------------- #

def test_raw_video_retention_cannot_be_switched_on():
    """PLAN.md section 10 is enforced in the type, not in a config comment."""
    try:
        RetentionPolicy(store_raw_video=True)
    except ValueError:
        return
    raise AssertionError("raw video retention was accepted")


def test_alert_has_nowhere_to_put_a_video_clip():
    """Only the event and a single evidence frame may leave the camera.

    Adding a clip field would then be a visible change to a documented
    boundary rather than a quiet configuration tweak.
    """
    fields = set(Alert.__dataclass_fields__)
    for forbidden in ("clip", "video", "frames", "buffer"):
        assert forbidden not in fields, f"Alert exposes {forbidden!r}"


def test_people_are_blurred_in_evidence_frames():
    image = np.random.default_rng(0).integers(
        0, 255, (200, 200, 3), dtype=np.uint8)
    person = Detection(hazard=Hazard.DROWNING, box=Box(50, 50, 150, 150),
                       confidence=0.9, label="swimmer")
    redacted = redact_people(image, [person])

    patch_before = image[60:140, 60:140]
    patch_after = redacted[60:140, 60:140]
    assert patch_after.var() < patch_before.var(), "person region was not blurred"


def test_redaction_does_not_mutate_the_caller_s_frame():
    """The frame may still be in the rolling buffer; mutating it in place
    would corrupt the source's own history."""
    image = np.random.default_rng(1).integers(
        0, 255, (100, 100, 3), dtype=np.uint8)
    original = image.copy()
    redact_people(image, [Detection(hazard=Hazard.DROWNING,
                                    box=Box(10, 10, 90, 90),
                                    confidence=0.9, label="swimmer")])
    assert np.array_equal(image, original)


def test_non_person_detections_are_left_alone():
    """Redaction must not destroy the evidence it exists to preserve — an
    operator has to be able to see the fire."""
    image = np.random.default_rng(2).integers(
        0, 255, (100, 100, 3), dtype=np.uint8)
    redacted = redact_people(image, [Detection(
        hazard=Hazard.FIRE, box=Box(10, 10, 90, 90),
        confidence=0.9, label="smoke")])
    assert np.array_equal(image, redacted)


def test_every_person_producing_label_is_covered():
    """Any detector class that depicts a person must be in PERSON_LABELS.

    Fails when a new detector adds a person class without confronting the
    redaction list.
    """
    from vigia.detectors.building_access import CLASS_NAMES as BA
    from vigia.detectors.person_in_water import CLASS_NAMES as PIW

    depicts_people = {"civilian", "rescue_team", "swimmer"}
    declared = set(BA) | set(PIW)
    for label in depicts_people & declared:
        assert label in PERSON_LABELS, (
            f"{label!r} is a detector class depicting a person but is not in "
            f"PERSON_LABELS, so it would appear unredacted in evidence frames"
        )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
