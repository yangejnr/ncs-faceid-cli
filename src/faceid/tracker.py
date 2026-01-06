from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float


class IoUTracker:
    """
    Lightweight tracker: matches detections to existing tracks via IoU.
    Good enough for a laptop MVP demo.
    """
    def __init__(self, iou_threshold: float = 0.3, ttl_seconds: float = 1.5):
        self.iou_threshold = float(iou_threshold)
        self.ttl_seconds = float(ttl_seconds)
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        now = time.time()
        # Drop stale tracks
        stale = [tid for tid, t in self.tracks.items() if (now - t.last_seen) > self.ttl_seconds]
        for tid in stale:
            del self.tracks[tid]

        assignments: Dict[int, Tuple[int, int, int, int]] = {}

        unmatched = set(range(len(bboxes)))
        # Greedy match: for each track, take best bbox by IoU
        for tid, t in list(self.tracks.items()):
            best_j = None
            best_score = 0.0
            for j in list(unmatched):
                s = iou(t.bbox, bboxes[j])
                if s > best_score:
                    best_score = s
                    best_j = j
            if best_j is not None and best_score >= self.iou_threshold:
                bbox = bboxes[best_j]
                t.bbox = bbox
                t.last_seen = now
                assignments[tid] = bbox
                unmatched.remove(best_j)

        # Create new tracks for unmatched detections
        for j in sorted(unmatched):
            bbox = bboxes[j]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(track_id=tid, bbox=bbox, last_seen=now)
            assignments[tid] = bbox

        return assignments
