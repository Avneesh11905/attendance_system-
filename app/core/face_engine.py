"""
Wraps InsightFace for face detection (SCRFD) and recognition (ArcFace).
Both models run through ONNX Runtime. The buffalo_l model pack gets
downloaded automatically the first time (~300MB).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from app.config import MODEL_NAME, DET_SIZE, SIMILARITY_THRESHOLD, MODEL_DIR
from app.utils.helpers import cosine_similarity


@dataclass
class FaceResult:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    landmarks: np.ndarray
    embedding: np.ndarray   # 512-D vector
    det_score: float


class FaceEngine:
    """
    Loads InsightFace models and provides detect + embed + match functionality.
    Call load_models() once at startup before using anything else.
    """

    def __init__(self):
        self._app = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_models(self):
        """Load SCRFD + ArcFace from the buffalo_l pack. Downloads on first run."""
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=MODEL_NAME,
            root=MODEL_DIR,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=DET_SIZE)
        self._is_loaded = True

    def detect_and_embed(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Run the full pipeline on a frame: detect faces, align them,
        and extract 512-D embeddings.
        Returns empty list if no faces found.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded yet — call load_models() first")

        faces = self._app.get(frame)
        results = []

        for face in faces:
            landmarks = face.kps
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106

            results.append(FaceResult(
                bbox=face.bbox,
                landmarks=landmarks,
                embedding=face.normed_embedding,
                det_score=float(face.det_score),
            ))

        return results

    def find_best_match(self, embedding, registered_users, threshold=SIMILARITY_THRESHOLD):
        """
        Compare an embedding against all registered users.
        Returns (user, score) if someone matches above threshold, else None.
        """
        if not registered_users:
            return None

        best_match = None
        best_score = -1.0

        for user in registered_users:
            score = cosine_similarity(embedding, user.embedding)
            if score > best_score:
                best_score = score
                best_match = user

        if best_match is not None and best_score >= threshold:
            return (best_match, best_score)

        return None
