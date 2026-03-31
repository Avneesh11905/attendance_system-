"""
Face Engine — Wraps InsightFace's FaceAnalysis for face detection and recognition.

Uses SCRFD (det_10g.onnx) for detection and ArcFace (w600k_r50.onnx) for
generating 512-dimensional face embeddings, both running on ONNX Runtime.

The buffalo_l model pack is automatically downloaded on first run (~300MB)
and cached in the face_models/ directory.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from app.config import MODEL_NAME, DET_SIZE, SIMILARITY_THRESHOLD, MODEL_DIR
from app.utils.helpers import cosine_similarity


@dataclass
class FaceResult:
    """Container for a single detected face's data."""
    bbox: np.ndarray          # Bounding box [x1, y1, x2, y2]
    landmarks: np.ndarray     # Facial keypoints (5-point or 106-point)
    embedding: np.ndarray     # 512-D ArcFace feature vector
    det_score: float          # Detection confidence (0-1)


class FaceEngine:
    """
    High-level wrapper around InsightFace's FaceAnalysis pipeline.

    Handles model loading, face detection, embedding extraction,
    and face matching against a database of registered users.

    Usage:
        engine = FaceEngine()
        engine.load_models()  # Call once at startup
        faces = engine.detect_and_embed(frame)
        match = engine.find_best_match(faces[0].embedding, registered_users)
    """

    def __init__(self):
        self._app = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_models(self):
        """
        Load the InsightFace model pack (SCRFD + ArcFace) via ONNX Runtime.

        This downloads the buffalo_l models on first run (~300MB).
        Uses CPU execution provider by default (ctx_id=-1).
        For GPU acceleration, change providers to ["CUDAExecutionProvider"].
        """
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=MODEL_NAME,
            root=MODEL_DIR,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=DET_SIZE)
        self._is_loaded = True
        print(f"[FaceEngine] Models loaded successfully: {MODEL_NAME}")
        print(f"[FaceEngine] Detection size: {DET_SIZE}")
        print(f"[FaceEngine] Provider: CPUExecutionProvider")

    def detect_and_embed(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Detect all faces in a frame and extract their embeddings.

        This runs the full pipeline:
        1. SCRFD face detection → bounding boxes + landmarks
        2. Face alignment using 5-point landmarks
        3. ArcFace embedding extraction → 512-D vector

        Args:
            frame: BGR image as numpy array (from OpenCV)

        Returns:
            List of FaceResult objects, one per detected face.
            Empty list if no faces detected.

        Raises:
            RuntimeError: If models haven't been loaded yet.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        faces = self._app.get(frame)

        results = []
        for face in faces:
            # InsightFace face objects have .kps (5 keypoints) and
            # optionally .landmark_2d_106 for detailed landmarks
            landmarks = face.kps
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106

            results.append(
                FaceResult(
                    bbox=face.bbox,
                    landmarks=landmarks,
                    embedding=face.normed_embedding,
                    det_score=float(face.det_score),
                )
            )

        return results

    def find_best_match(
        self,
        embedding: np.ndarray,
        registered_users: list,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> Optional[tuple]:
        """
        Compare a face embedding against all registered users to find
        the best match.

        Uses cosine similarity — ArcFace embeddings are L2-normalized,
        so cosine similarity is equivalent to dot product.

        Args:
            embedding: 512-D face embedding to match
            registered_users: List of User objects (must have .embedding attribute)
            threshold: Minimum similarity score for a valid match (default: 0.4)

        Returns:
            Tuple of (User, similarity_score) if a match is found above threshold.
            None if no match meets the threshold.
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
