"""Water level estimation and rate-of-rise — the anticipatory flood signal.

NOT A NOVEL CONTRIBUTION. Read this before describing it as one.

Segmenting water in a street scene is a solved problem: published CNN
approaches report F1 above 0.9 on real surveillance footage. The gap most of
that literature leaves is anticipatory — it detects inundation *after* it has
occurred. Knowing a street is flooded is useful; knowing it is filling, and how
fast, is what buys evacuation time.

We initially treated closing that gap as VIGÍA's one novel component. It is
not. Choi, Kim, Win Aung and Park, "Real-time anticipatory urban flood warning
using CCTV and Page-Hinkley change detection", Developments in the Built
Environment 25 (2026), article 100866, target the same gap on the same
infrastructure, with short-term trend analysis producing an estimated
time-to-threshold — the same capability as this module.

So this module implements two trend mechanisms and measures them against each
other, rather than claiming either: the sliding-window least-squares fit below,
and a Page-Hinkley change detector at the end of the file following Choi et al.
VIGÍA's contribution is the multi-hazard pipeline these sit inside, not the
flood trend signal itself.

During the 2024 Valencia DANA the difference between those two facts was
measured in lives. AEMET's red warning was issued hours before the worst of the
rainfall; the alert that reached people's phones came late that evening. A
camera watching water climb a kerb had the information the whole time.

Two signals are produced, deliberately:

  * `area_fraction` — the proportion of the frame that is water. Needs no
    calibration at all, so it works on any camera the moment it is connected.
  * `level` — where the water boundary crosses a per-camera reference line.
    Needs a one-off calibration but is physically meaningful and comparable
    between cameras.

Rate of rise is computed for whichever signals are available by least-squares
fit over a sliding time window, rather than by differencing consecutive frames.
Frame-to-frame differencing on a noisy segmentation mask produces a rate
estimate dominated by segmentation jitter; a fit over a window is stable enough
to threshold on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np


#: Minimum readings needed before a least-squares slope is trustworthy, and
#: the number of sampling intervals the fit window must be able to hold.
MIN_SAMPLES_FOR_FIT = 4


class Trend(str, Enum):
    """What the water is doing. This, not the absolute level, is the alert."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    UNKNOWN = "unknown"      # not enough history yet


@dataclass(frozen=True)
class ReferenceLine:
    """A calibrated vertical line in image space used to read water height.

    `top` and `bottom` are (x, y) pixel coordinates. `metres` is the real-world
    height difference between them, if known — when supplied, levels are
    reported in metres instead of as a fraction.

    Choosing the line is a human decision made once per camera: a lamp post, a
    building edge, a flood gauge. It is deliberately not automated; a wrong
    automatic choice would silently corrupt every reading from that camera.
    """

    top: tuple[float, float]
    bottom: tuple[float, float]
    metres: Optional[float] = None

    def sample_points(self, count: int = 200) -> np.ndarray:
        """Points from bottom to top, so index 0 is the lowest point."""
        x_top, y_top = self.top
        x_bottom, y_bottom = self.bottom
        t = np.linspace(0.0, 1.0, count)
        xs = x_bottom + (x_top - x_bottom) * t
        ys = y_bottom + (y_top - y_bottom) * t
        return np.stack([xs, ys], axis=1)


@dataclass
class LevelReading:
    """One frame's water measurement."""

    timestamp: float
    area_fraction: float
    level_fraction: Optional[float] = None    # 0 = dry, 1 = line fully submerged
    level_metres: Optional[float] = None

    @property
    def has_level(self) -> bool:
        return self.level_fraction is not None


@dataclass
class RiseEstimate:
    """Rate of change, fitted over a window of readings."""

    trend: Trend
    area_rate_per_min: float = 0.0            # frame fraction per minute
    level_rate_per_min: Optional[float] = None    # line fraction per minute
    metres_per_min: Optional[float] = None
    samples: int = 0
    window_seconds: float = 0.0
    r_squared: float = 0.0                    # how linear the rise is

    @property
    def confident(self) -> bool:
        """Enough samples over enough time, and a reasonably linear fit.

        A poor fit usually means the segmentation is unstable rather than that
        the water is doing something complicated, so we decline to call it.
        """
        return (self.samples >= MIN_SAMPLES_FOR_FIT
                and self.window_seconds > 0
                and self.r_squared >= 0.5)


def _fit_rate(times: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope (units per second) and R² for a series."""
    if len(times) < 2:
        return 0.0, 0.0

    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    t = t - t[0]

    if np.allclose(t, t[0]) or np.allclose(y, y[0]):
        return 0.0, 1.0 if np.allclose(y, y[0]) else 0.0

    slope, intercept = np.polyfit(t, y, 1)
    predicted = slope * t + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), max(0.0, r_squared)


class WaterLevelTracker:
    """Turns a stream of water masks into a level and a rate of rise.

    Args:
        reference_line: optional per-camera calibration. Without it only the
            area signal is produced, which is still usable.
        window_seconds: how much history the rate is fitted over. 300 s suits
            urban flash flooding, where the DANA's dangerous rises happened over
            minutes rather than hours.
        rising_threshold: area fraction per minute above which we call it
            rising. Set from data, not intuition — see calibrate_threshold().
    """

    def __init__(
        self,
        reference_line: Optional[ReferenceLine] = None,
        *,
        window_seconds: float = 300.0,
        rising_threshold: float = 0.005,
        max_readings: int = 512,
    ) -> None:
        self.reference_line = reference_line
        self.window_seconds = window_seconds
        self.rising_threshold = rising_threshold
        self.readings: deque[LevelReading] = deque(maxlen=max_readings)

    # ------------------------------------------------------------------ #

    def measure(self, water_mask: np.ndarray, timestamp: float) -> LevelReading:
        """Measure one frame. `water_mask` is boolean or 0/1, shape (H, W)."""
        mask = water_mask.astype(bool)
        area_fraction = float(mask.mean()) if mask.size else 0.0

        level_fraction: Optional[float] = None
        if self.reference_line is not None:
            level_fraction = self._level_along_line(mask)

        level_metres = None
        if level_fraction is not None and self.reference_line is not None:
            if self.reference_line.metres is not None:
                level_metres = level_fraction * self.reference_line.metres

        reading = LevelReading(
            timestamp=timestamp,
            area_fraction=area_fraction,
            level_fraction=level_fraction,
            level_metres=level_metres,
        )
        self.readings.append(reading)
        return reading

    def _level_along_line(self, mask: np.ndarray) -> Optional[float]:
        """Fraction of the reference line, from the bottom, that is water.

        We walk from the bottom up and stop at the first sustained dry stretch
        rather than taking the highest water pixel anywhere on the line. A
        reflection or a puddle further up the line would otherwise read as a
        far higher water level than reality.
        """
        assert self.reference_line is not None
        height, width = mask.shape[:2]
        points = self.reference_line.sample_points(count=200)

        wet = np.zeros(len(points), dtype=bool)
        for index, (x, y) in enumerate(points):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                wet[index] = bool(mask[yi, xi])

        if not wet.any():
            return 0.0

        # Walk up from the bottom; stop after 5 consecutive dry samples (2.5% of
        # the line), which tolerates segmentation speckle without jumping gaps.
        dry_run = 0
        last_wet = -1
        for index, is_wet in enumerate(wet):
            if is_wet:
                last_wet = index
                dry_run = 0
            else:
                dry_run += 1
                if dry_run >= 5:
                    break

        if last_wet < 0:
            return 0.0
        return (last_wet + 1) / len(points)

    # ------------------------------------------------------------------ #

    def estimate_rise(self, now: Optional[float] = None) -> RiseEstimate:
        """Fit the rate of change over the trailing window."""
        if not self.readings:
            return RiseEstimate(trend=Trend.UNKNOWN)

        now = self.readings[-1].timestamp if now is None else now

        # The window must adapt to the camera's sampling interval.
        #
        # Found on real data: a fixed 300 s window silently failed on a creek
        # camera sampled every 15 minutes — the window could never contain two
        # frames, so the fit reported UNKNOWN for seven hours while the water
        # rose by 10% of the frame. Page-Hinkley, being sample-based rather
        # than time-based, was unaffected. A time window alone is the wrong
        # abstraction when the sampling rate is a property of the deployment.
        effective_window = self.window_seconds
        if len(self.readings) >= 3:
            gaps = [
                self.readings[i + 1].timestamp - self.readings[i].timestamp
                for i in range(len(self.readings) - 1)
            ]
            gaps = sorted(g for g in gaps if g > 0)
            if gaps:
                median_gap = gaps[len(gaps) // 2]
                # Always admit at least MIN_SAMPLES_FOR_FIT intervals.
                effective_window = max(
                    effective_window, median_gap * MIN_SAMPLES_FOR_FIT
                )

        window = [r for r in self.readings if now - r.timestamp <= effective_window]
        if len(window) < 2:
            return RiseEstimate(trend=Trend.UNKNOWN, samples=len(window))

        times = [r.timestamp for r in window]
        area_slope, area_r2 = _fit_rate(times, [r.area_fraction for r in window])
        area_rate_per_min = area_slope * 60.0

        level_rate_per_min = None
        metres_per_min = None
        level_r2 = 0.0
        levelled = [r for r in window if r.has_level]
        if len(levelled) >= 2:
            level_slope, level_r2 = _fit_rate(
                [r.timestamp for r in levelled],
                [r.level_fraction for r in levelled],  # type: ignore[misc]
            )
            level_rate_per_min = level_slope * 60.0
            if self.reference_line is not None and self.reference_line.metres:
                metres_per_min = level_rate_per_min * self.reference_line.metres

        # Prefer the calibrated signal for the trend decision when we have one:
        # area fraction moves with camera perspective, level does not.
        primary_rate = (
            level_rate_per_min if level_rate_per_min is not None else area_rate_per_min
        )
        primary_r2 = level_r2 if level_rate_per_min is not None else area_r2

        if abs(primary_rate) < self.rising_threshold:
            trend = Trend.STABLE
        elif primary_rate > 0:
            trend = Trend.RISING
        else:
            trend = Trend.FALLING

        return RiseEstimate(
            trend=trend,
            area_rate_per_min=area_rate_per_min,
            level_rate_per_min=level_rate_per_min,
            metres_per_min=metres_per_min,
            samples=len(window),
            window_seconds=float(times[-1] - times[0]),
            r_squared=primary_r2,
        )

    def time_to_threshold(self, target_level: float) -> Optional[float]:
        """Seconds until the level reaches `target_level` at the current rate.

        This is the number an operator actually wants: not "the street is
        flooding" but "this underpass is impassable in eleven minutes".
        Returns None when the water is not rising or there is no calibrated
        level to extrapolate from.
        """
        estimate = self.estimate_rise()
        if estimate.trend is not Trend.RISING or not estimate.confident:
            return None
        if estimate.level_rate_per_min is None or estimate.level_rate_per_min <= 0:
            return None

        current = self.readings[-1].level_fraction
        if current is None or current >= target_level:
            return 0.0

        minutes = (target_level - current) / estimate.level_rate_per_min
        return minutes * 60.0

    def reset(self) -> None:
        self.readings.clear()


# --------------------------------------------------------------------------- #
# Page-Hinkley change detection
# --------------------------------------------------------------------------- #
#
# PRIOR ART, ACKNOWLEDGED. Choi, Kim, Win Aung and Park, "Real-time anticipatory
# urban flood warning using CCTV and Page-Hinkley change detection",
# Developments in the Built Environment 25 (2026), article 100866, apply exactly
# this idea to exactly this problem: anticipatory urban flood warning from
# existing CCTV, using Page-Hinkley to spot incipient water-level rise while
# suppressing false alarms, with short-term trend analysis producing an
# estimated time-to-threshold.
#
# That is the same capability as WaterLevelTracker above, published
# independently. VIGÍA therefore does NOT claim rate-of-rise for anticipatory
# flood warning as a novel contribution. What we do instead is implement both
# mechanisms and measure them against each other on the same data, and place the
# result inside a multi-hazard pipeline — which is the part that is ours.
#
# Page-Hinkley is a classical sequential change-point test. It accumulates the
# signed deviation of a signal from its running mean and alarms when the
# accumulated deviation departs far enough from its own running minimum. Its
# character is different from a windowed least-squares fit: it responds to the
# ONSET of a change quickly, whereas the fit estimates the RATE of an ongoing
# one more stably. They are complementary rather than competing, which is why
# both are available here.

@dataclass
class PageHinkleyState:
    """Internal state of the Page-Hinkley test."""

    cumulative: float = 0.0
    minimum: float = 0.0
    mean: float = 0.0
    samples: int = 0
    alarmed: bool = False
    alarm_time: Optional[float] = None


class PageHinkleyDetector:
    """Sequential change-point detection on a scalar signal.

    Args:
        delta: tolerated drift. Changes slower than this are treated as noise
            rather than a trend, so it sets the sensitivity floor.
        threshold: how far the accumulated deviation must depart from its own
            minimum before alarming. Higher means later but surer.
        min_samples: no alarm before this many observations, so a single early
            outlier cannot trigger one.

    CALIBRATION MATTERS MORE THAN THE ALGORITHM. Page-Hinkley accumulates
    deviations, so on a noisy signal the accumulator random-walks and will
    eventually cross any threshold that is small relative to the noise. Our
    first defaults (delta 0.002, threshold 0.02) looked reasonable and produced
    a 92% false-alarm rate on *stable* water with realistic segmentation noise.

    Measured over 200 trials of 40 samples of stable water, false-alarm rate by
    parameter set and noise level (eval/results/page_hinkley_calibration.json):

        delta  threshold | sigma=0.005  sigma=0.010  sigma=0.020
        0.002      0.02  |      15.0%        92.0%       100.0%
        0.002      0.05  |       0.0%        23.0%        90.5%
        0.005      0.05  |       0.0%         2.0%        74.5%
        0.005      0.10  |       0.0%         0.0%        16.0%   <- default
        0.010      0.10  |       0.0%         0.0%         2.0%

    And detection latency for a genuine rise, in samples:

        delta  threshold | 0.5%/frame  1%/frame  3%/frame
        0.005      0.10  |         11         7         4   <- default
        0.010      0.10  |         15         8         4

    The defaults below trade a little latency for zero false alarms at typical
    segmentation noise. A camera with noisier segmentation needs a higher
    threshold, and that should be measured per camera rather than assumed.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 0.10,
                 min_samples: int = 5) -> None:
        self.delta = delta
        self.threshold = threshold
        self.min_samples = min_samples
        self.state = PageHinkleyState()

    def update(self, value: float, timestamp: float) -> bool:
        """Feed one observation. Returns True on the frame an increase is first
        detected, and stays latched thereafter until reset."""
        state = self.state
        state.samples += 1
        # Running mean, updated incrementally.
        state.mean += (value - state.mean) / state.samples

        # Accumulate deviation above the mean, less the tolerated drift.
        state.cumulative += value - state.mean - self.delta
        state.minimum = min(state.minimum, state.cumulative)

        if state.samples < self.min_samples:
            return False

        if not state.alarmed and (state.cumulative - state.minimum) > self.threshold:
            state.alarmed = True
            state.alarm_time = timestamp
            return True
        return False

    @property
    def statistic(self) -> float:
        """Current test statistic — how far above its minimum the accumulator
        has climbed. Useful to display as a rising-risk gauge."""
        return self.state.cumulative - self.state.minimum

    def reset(self) -> None:
        self.state = PageHinkleyState()
