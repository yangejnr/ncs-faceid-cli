from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import numpy as np

# InsightFace
from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray  # normalized float32 (D,)
    quality: float         # heuristic quality score


class FaceEngine:
    """
    Wraps InsightFace FaceAnalysis:
      - detect faces
      - return embeddings (ArcFace)
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)):
        # Store models in repo to make demo portable.
        os.environ.setdefault("INSIGHTFACE_HOME", os.path.abspath(".models"))

        self.model_name = model_name
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Embed dim depends on model; buffalo_l is typically 512.
        self.model_version = f"insightface:{model_name}"

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-12
        return v / n

    @staticmethod
    def _quality(bbox: tuple[int, int, int, int], det_score: float) -> float:
        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = float(w * h)
        # Simple heuristic: detection confidence * sqrt(area)
        return float(det_score * (area ** 0.5))

    def detect(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        faces = self.app.get(bgr_image)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = [int(x) for x in f.bbox]
            det_score = float(getattr(f, "det_score", 0.0))
            emb = self._normalize(np.asarray(f.embedding, dtype=np.float32))
            q = self._quality((x1, y1, x2, y2), det_score)
            out.append(DetectedFace(bbox=(x1, y1, x2, y2), det_score=det_score, embedding=emb, quality=q))
        return out
