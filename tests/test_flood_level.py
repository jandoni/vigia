"""Tests for the water level and rate-of-rise signal.

This is the component intended to be novel, so its behaviour is pinned by
tests that assert the properties we will claim: that it detects a rise before
inundation is complete, that it is not fooled by segmentation noise, and that
its time-to-threshold estimate is arithmetically sound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigia.flood.level import (  # noqa: E402
    PageHinkleyDetector, ReferenceLine, Trend, WaterLevelTracker,
)

HEIGHT, WIDTH = 400, 600


def water_mask(fill_fraction: float) -> np.ndarray:
    """A mask flooded from the bottom up to `fill_fraction` of frame height."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    rows = int(HEIGHT * fill_fraction)
    if rows > 0:
        mask[HEIGHT - rows:, :] = True
    return mask


# A vertical line down the middle of the frame, 3 m tall in the real world.
LINE = ReferenceLine(top=(300.0, 0.0), bottom=(300.0, float(HEIGHT - 1)), metres=3.0)


def test_area_fraction_tracks_flooding():
    tracker = WaterLevelTracker()
    reading = tracker.measure(water_mask(0.25), timestamp=0.0)
    assert abs(reading.area_fraction - 0.25) < 0.01


def test_level_along_reference_line():
    tracker = WaterLevelTracker(LINE)
    reading = tracker.measure(water_mask(0.5), timestamp=0.0)
    assert reading.level_fraction is not None
    assert abs(reading.level_fraction - 0.5) < 0.05
    # 3 m line, half submerged
    assert reading.level_metres is not None
    assert abs(reading.level_metres - 1.5) < 0.2


def test_detects_rise_before_inundation_is_complete():
    """The whole point: call it while the water is still climbing.

    Water rises from 10% to 30% of frame height over five minutes. The trend
    must read RISING well before the scene is fully flooded.
    """
    tracker = WaterLevelTracker(LINE, window_seconds=600)
    for step in range(11):
        timestamp = step * 30.0
        tracker.measure(water_mask(0.10 + 0.02 * step), timestamp)

    estimate = tracker.estimate_rise()
    assert estimate.trend is Trend.RISING, f"expected RISING, got {estimate.trend}"
    assert estimate.confident, "should be confident over 5 minutes of steady rise"
    assert estimate.level_rate_per_min is not None
    assert estimate.level_rate_per_min > 0
    assert estimate.r_squared > 0.9, "a linear rise should fit almost perfectly"


def test_stable_water_is_not_reported_as_rising():
    """Standing water must not raise a rise alert, however deep it is."""
    tracker = WaterLevelTracker(LINE, window_seconds=600)
    for step in range(12):
        tracker.measure(water_mask(0.40), timestamp=step * 30.0)

    estimate = tracker.estimate_rise()
    assert estimate.trend is Trend.STABLE


def test_segmentation_noise_does_not_create_a_false_rise():
    """Mask jitter around a constant level must not read as a trend.

    This is why the rate is fitted over a window instead of differenced between
    consecutive frames: differencing turns segmentation noise into a rate.
    """
    rng = np.random.default_rng(0)
    tracker = WaterLevelTracker(LINE, window_seconds=600)
    for step in range(20):
        jitter = float(rng.normal(0.0, 0.02))
        tracker.measure(water_mask(max(0.0, 0.40 + jitter)), timestamp=step * 30.0)

    estimate = tracker.estimate_rise()
    assert estimate.trend is Trend.STABLE, (
        f"noise around a constant level read as {estimate.trend} "
        f"(rate {estimate.level_rate_per_min})"
    )


def test_falling_water_is_distinguished_from_rising():
    tracker = WaterLevelTracker(LINE, window_seconds=600)
    for step in range(11):
        tracker.measure(water_mask(0.50 - 0.03 * step), timestamp=step * 30.0)

    assert tracker.estimate_rise().trend is Trend.FALLING


def test_time_to_threshold_is_arithmetically_sound():
    """The operational number: minutes until the water reaches a danger line."""
    tracker = WaterLevelTracker(LINE, window_seconds=900)
    # Rise 2% of the line per 30 s => 4% per minute.
    for step in range(11):
        tracker.measure(water_mask(0.20 + 0.02 * step), timestamp=step * 30.0)

    seconds = tracker.time_to_threshold(target_level=0.80)
    assert seconds is not None
    # Currently at 0.40, rising 4%/min, so 0.40 to go => ~10 minutes.
    assert 7 * 60 < seconds < 14 * 60, f"expected ~10 min, got {seconds / 60:.1f} min"


def test_no_estimate_without_enough_history():
    tracker = WaterLevelTracker(LINE)
    tracker.measure(water_mask(0.3), timestamp=0.0)
    assert tracker.estimate_rise().trend is Trend.UNKNOWN
    assert tracker.time_to_threshold(0.8) is None


def test_reflection_above_the_waterline_does_not_inflate_level():
    """A detached wet patch high on the line must not read as deep water.

    Reflections, wet walls and puddles on balconies all produce this. Taking
    the highest water pixel on the line would report the building as submerged.
    """
    tracker = WaterLevelTracker(LINE)
    mask = water_mask(0.20)
    mask[50:70, 290:310] = True          # detached patch near the top of the line
    reading = tracker.measure(mask, timestamp=0.0)

    assert reading.level_fraction is not None
    assert reading.level_fraction < 0.35, (
        f"detached patch inflated the level to {reading.level_fraction:.2f}"
    )



# --------------------------------------------------------------------------- #
# Page-Hinkley — the mechanism published by Choi et al. (2026) for this problem.
# Implemented here so the two approaches can be compared rather than one being
# claimed. See the note at the top of vigia/flood/level.py.
# --------------------------------------------------------------------------- #

def test_page_hinkley_alarms_on_a_rise():
    detector = PageHinkleyDetector(min_samples=4)  # calibrated defaults
    alarmed_at = None
    for step in range(20):
        value = 0.20 + 0.01 * step          # steady rise
        if detector.update(value, timestamp=step * 30.0) and alarmed_at is None:
            alarmed_at = step
    assert alarmed_at is not None, "a steady rise must eventually alarm"
    assert alarmed_at >= 3, "must respect min_samples (4th sample = index 3)"


def test_page_hinkley_stays_quiet_on_stable_water():
    """Standing water, with noise, must not alarm. This is the property that
    makes the mechanism usable — an alarm on every windy afternoon is useless."""
    rng = np.random.default_rng(0)
    detector = PageHinkleyDetector(min_samples=4)  # calibrated defaults
    alarms = 0
    for step in range(40):
        value = 0.35 + float(rng.normal(0.0, 0.01))
        alarms += int(detector.update(value, timestamp=step * 30.0))
    assert alarms == 0, "noise around a constant level triggered a change alarm"


def test_page_hinkley_reacts_faster_than_the_least_squares_fit():
    """The reason both mechanisms exist.

    Page-Hinkley responds to the ONSET of a change; the windowed fit needs
    enough samples inside its window to produce a confident slope. On an abrupt
    rise the change detector should call it no later than the fit does.
    """
    tracker = WaterLevelTracker(window_seconds=600, rising_threshold=0.005)
    detector = PageHinkleyDetector(min_samples=4)  # calibrated defaults

    ph_alarm = ls_call = None
    for step in range(24):
        # Flat for 8 samples, then a sharp rise.
        fill = 0.20 if step < 8 else 0.20 + 0.03 * (step - 8)
        timestamp = step * 30.0
        tracker.measure(water_mask(min(fill, 0.95)), timestamp)

        if detector.update(tracker.readings[-1].area_fraction, timestamp) and ph_alarm is None:
            ph_alarm = step
        estimate = tracker.estimate_rise(timestamp)
        if ls_call is None and estimate.trend is Trend.RISING and estimate.confident:
            ls_call = step

    assert ph_alarm is not None, "Page-Hinkley missed an abrupt rise"
    assert ls_call is not None, "the least-squares fit missed an abrupt rise"
    assert ph_alarm <= ls_call, (
        f"Page-Hinkley ({ph_alarm}) should not be slower than the fit ({ls_call})"
    )


def test_page_hinkley_resets_cleanly():
    detector = PageHinkleyDetector(min_samples=4)  # calibrated defaults
    for step in range(20):
        detector.update(0.2 + 0.01 * step, timestamp=step * 30.0)
    assert detector.state.alarmed

    detector.reset()
    assert not detector.state.alarmed
    assert detector.state.samples == 0
    assert detector.statistic == 0.0


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
