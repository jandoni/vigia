"""Behavioural tests for the temporal validator.

These are not incidental unit tests — the validator is the part of VIGÍA we
claim as a contribution, so its behaviour has to be pinned down and stay
pinned. Each test asserts one property we will state in the dossier.

Run:  python -m pytest tests/ -v      (or)  python tests/test_validator.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigia.types import Box, Detection, Hazard  # noqa: E402
from vigia.validator.cascade import TemporalValidator, ValidatorConfig  # noqa: E402

FRAME = np.zeros((1024, 1024, 3), dtype=np.uint8)


def make_detection(x: int, y: int, confidence: float = 0.5, size: int = 60,
                   frame: int = 0, timestamp: float = 0.0) -> Detection:
    return Detection(
        hazard=Hazard.FIRE,
        box=Box(x, y, x + size, y + size),
        confidence=confidence,
        frame_index=frame,
        timestamp=timestamp,
        camera_id="test-cam",
    )


def test_persistent_detection_is_confirmed():
    """A hazard that stays put is confirmed once it has been seen min_frames
    times — and not before. This is the property the whole design rests on."""
    validator = TemporalValidator(
        Hazard.FIRE, ValidatorConfig(min_frames=3, enable_cooldown=False)
    )
    confirmations = []
    for frame in range(6):
        events = validator.process(
            [make_detection(400, 400, 0.6, frame=frame, timestamp=frame / 5)],
            frame, frame / 5, FRAME, "test-cam",
        )
        confirmations.append(len(events))

    assert confirmations[:3] == [0, 0, 1], (
        f"expected first confirmation on the third sighting, got {confirmations}"
    )
    assert validator.confirmed_count == 4


def test_flickering_noise_is_never_confirmed():
    """Detections that appear at unrelated places on each frame never
    accumulate persistence. This is the false-positive mode the validator
    exists to kill."""
    validator = TemporalValidator(
        Hazard.FIRE, ValidatorConfig(min_frames=3, enable_cooldown=False)
    )
    rng = np.random.default_rng(0)
    for frame in range(20):
        x, y = (int(v) for v in rng.integers(0, 900, 2))
        validator.process(
            [make_detection(x, y, 0.6, frame=frame, timestamp=frame / 5)],
            frame, frame / 5, FRAME, "test-cam",
        )

    assert validator.confirmed_count == 0


def test_cooldown_collapses_a_continuing_event():
    """One fire burning for 30 s must not produce 30 alerts. Technically
    correct, operationally useless — an operator stops reading after the
    third."""
    validator = TemporalValidator(
        Hazard.FIRE,
        ValidatorConfig(min_frames=2, enable_cooldown=True, cooldown_seconds=10),
    )
    alerts = 0
    for frame in range(30):
        alerts += len(
            validator.process(
                [make_detection(400, 400, 0.6, frame=frame, timestamp=float(frame))],
                frame, float(frame), FRAME, "test-cam",
            )
        )

    assert 2 <= alerts <= 4, f"expected ~3 alerts across 30 s, got {alerts}"


def test_size_level_was_removed_and_stays_removed():
    """A size-plausibility level existed and was deliberately removed.

    Measured, it rejected nothing on wildfire and nothing on traffic. On flood
    it was actively dangerous: real ATLANTIS flood masks cover a median 32% of
    the frame and severe cases exceed 60%, so a plausible max-area default
    silently discarded the worst floods — the more severe the disaster, the
    more likely the alert vanished.

    This test exists so nobody reintroduces it without re-reading that.
    """
    config = ValidatorConfig()
    assert not hasattr(config, "enable_size")
    assert "size" not in TemporalValidator.LEVEL_NAMES

    # A detection covering 80% of the frame must survive to the validator.
    validator = TemporalValidator(
        Hazard.FIRE,
        ValidatorConfig(enable_persistence=False, enable_cooldown=False),
    )
    huge = Detection(hazard=Hazard.FIRE, box=Box(0, 0, 950, 900),
                     confidence=0.9, camera_id="test-cam")
    validator.process([huge], 0, 0.0, FRAME, "test-cam")
    assert huge.rejected_by is None, (
        "a frame-filling hazard was rejected — this is the flood failure mode"
    )


def test_confidence_level_rejects_low_scores():
    validator = TemporalValidator(
        Hazard.FIRE,
        ValidatorConfig(min_confidence=0.5, enable_persistence=False,
                        enable_cooldown=False),
    )
    weak = make_detection(400, 400, 0.30)
    strong = make_detection(400, 400, 0.70)
    validator.process([weak, strong], 0, 0.0, FRAME, "test-cam")

    assert weak.rejected_by == "confidence"
    assert strong.rejected_by is None


def test_levels_can_be_disabled_independently():
    """Required for the ablation: we must be able to report what each level
    contributes on its own."""
    config = ValidatorConfig(
        enable_confidence=True,
        enable_persistence=False, enable_cooldown=False,
    )
    validator = TemporalValidator(Hazard.FIRE, config)
    assert validator.enabled_levels == ["confidence"]


def test_reset_clears_state_between_videos():
    """Without this, tracks from one clip leak into the next and silently
    corrupt evaluation."""
    validator = TemporalValidator(Hazard.FIRE, ValidatorConfig(min_frames=3))
    for frame in range(5):
        validator.process(
            [make_detection(400, 400, 0.6, frame=frame, timestamp=frame / 5)],
            frame, frame / 5, FRAME, "test-cam",
        )
    assert validator.confirmed_count > 0

    validator.reset()
    assert validator.confirmed_count == 0
    assert validator.tracker.tracks == []
    assert all(s.seen == 0 for s in validator.stats.values())

    # A single detection right after a reset must not be instantly confirmed.
    events = validator.process(
        [make_detection(400, 400, 0.6, frame=0, timestamp=0.0)],
        0, 0.0, FRAME, "test-cam",
    )
    assert events == []


def test_stats_account_for_every_detection():
    """The signal-chain display and the ablation table both read these
    counters, so they must not lose detections."""
    validator = TemporalValidator(Hazard.FIRE, ValidatorConfig(min_frames=3))
    rng = np.random.default_rng(1)
    total = 0
    for frame in range(40):
        detections = [make_detection(400, 400, 0.6, frame=frame, timestamp=frame / 5)]
        detections += [
            make_detection(int(rng.integers(0, 900)), int(rng.integers(0, 900)),
                           float(rng.uniform(0.05, 0.9)), frame=frame, timestamp=frame / 5)
            for _ in range(3)
        ]
        total += len(detections)
        validator.process(detections, frame, frame / 5, FRAME, "test-cam")

    stats = validator.stats
    assert stats["confidence"].seen == total
    # Each level sees exactly what the previous one passed.
    assert stats["persistence"].seen == stats["confidence"].passed
    assert stats["cooldown"].seen == stats["persistence"].passed
    assert validator.confirmed_count == stats["cooldown"].passed


def test_colour_level_was_removed_and_stays_removed():
    """The colour prior was measured and removed (eval/run_colour_eval.py):
    fog and smoke are both grey, so on curated hard negatives it bought
    false-alarm rate only by discarding real smoke, and inside the cascade a
    strict cutoff lost an ignition sequence while no cutoff improved anything.
    Like the size level, it must not quietly return."""
    config_fields = set(ValidatorConfig.__dataclass_fields__)
    assert not any("colour" in field or "color" in field or "hue" in field
                   or "saturation" in field for field in config_fields), (
        "a colour/appearance field has reappeared on ValidatorConfig; "
        "see eval/results/colour_prior.json before reintroducing it")
    assert "colour" not in TemporalValidator.LEVEL_NAMES
    assert not hasattr(TemporalValidator, "_check_colour")


def test_validator_contains_no_hazard_specific_code():
    """The hazard-agnostic claim, enforced.

    VIGÍA's contribution is a validator that is *configured* per hazard, never
    *specialised* per hazard. If someone adds `if hazard == Hazard.FIRE` to the
    cascade to fix a fire bug, the claim in the dossier quietly becomes false.
    This test makes that impossible to do by accident.
    """
    source = (Path(__file__).resolve().parent.parent
              / "vigia" / "validator" / "cascade.py").read_text(encoding="utf-8")

    tree = ast.parse(source)
    offences: list[str] = []

    hazard_names = {h.name for h in Hazard} | {h.value for h in Hazard}

    for node in ast.walk(tree):
        # A comparison against a specific hazard is the thing we forbid.
        if isinstance(node, ast.Compare):
            for side in [node.left, *node.comparators]:
                if isinstance(side, ast.Attribute) and side.attr in hazard_names:
                    offences.append(f"line {node.lineno}: compares against Hazard.{side.attr}")
                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                    if side.value.lower() in {h.value for h in Hazard}:
                        offences.append(
                            f"line {node.lineno}: compares against literal {side.value!r}"
                        )

    assert not offences, (
        "cascade.py contains hazard-specific branching:\n  "
        + "\n  ".join(offences)
        + "\nEverything hazard-specific belongs in ValidatorConfig or the detector."
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
