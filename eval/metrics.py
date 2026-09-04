"""Detection metrics.

Two levels, kept strictly separate because they answer different questions:

  * Box level  — did we localise the hazard? Standard IoU-matched P/R/F.
  * Image level — did we raise an alarm on this frame at all? This is the
    level operators actually care about, and it is where the false-alarm rate
    that motivates the whole project is measured.

F2 rather than F1 is the headline for VIGÍA: recall matters more than
precision for life safety. Borrowed from RipVIS (arXiv 2504.01128), which
makes the same argument for beach monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from vigia.types import Box


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta. beta>1 weights recall higher; beta=2 weights it 4x."""
    b2 = beta * beta
    denominator = b2 * precision + recall
    if denominator <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denominator


@dataclass
class CountResult:
    """Raw confusion counts plus the rates derived from them."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        return fbeta(self.precision, self.recall, beta=1.0)

    @property
    def f2(self) -> float:
        return fbeta(self.precision, self.recall, beta=2.0)

    @property
    def false_alarm_rate(self) -> float:
        """Fraction of true-negative frames on which we wrongly alarmed.

        This is the number the temporal validator exists to reduce, and the
        one the reference literature reports moving from 52% to 4%.
        """
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            precision=round(self.precision, 4),
            recall=round(self.recall, 4),
            f1=round(self.f1, 4),
            f2=round(self.f2, 4),
            false_alarm_rate=round(self.false_alarm_rate, 4),
        )
        return data


def match_boxes(
    predictions: Sequence[Box],
    ground_truth: Sequence[Box],
    iou_threshold: float = 0.3,
) -> tuple[int, int, int]:
    """Greedily match predictions to ground truth by IoU.

    Returns (true_positives, false_positives, false_negatives). Predictions
    must already be sorted by descending confidence — each ground-truth box can
    only be claimed once, so ordering decides who claims it.

    An IoU threshold of 0.3 rather than the usual 0.5 is deliberate: a smoke
    plume has no crisp boundary, and annotators disagree about where it ends.
    Demanding 0.5 measures annotation convention more than detection quality.
    """
    claimed: set[int] = set()
    true_positives = 0

    for prediction in predictions:
        best_iou, best_index = 0.0, -1
        for index, truth in enumerate(ground_truth):
            if index in claimed:
                continue
            iou = prediction.iou(truth)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index >= 0 and best_iou >= iou_threshold:
            claimed.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def parse_yolo_label(text: str, image_width: int, image_height: int) -> list[Box]:
    """Parse YOLO-format label text into pixel-space boxes.

    Each line is `class cx cy w h`, all normalised to [0, 1].
    """
    boxes: list[Box] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(p) for p in parts[:5])
        bw, bh = w * image_width, h * image_height
        px, py = cx * image_width, cy * image_height
        boxes.append(Box(px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2))
    return boxes


def aggregate(results: Iterable[CountResult]) -> CountResult:
    total = CountResult()
    for result in results:
        total.true_positives += result.true_positives
        total.false_positives += result.false_positives
        total.false_negatives += result.false_negatives
        total.true_negatives += result.true_negatives
    return total
