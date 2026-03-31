"""
Utility helpers for image conversion, embedding serialization,
and face bounding box drawing.
"""

import io
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk


def frame_to_ctkimage(frame: np.ndarray, width: int, height: int) -> ctk.CTkImage:
    """
    Convert an OpenCV BGR frame to a CTkImage for display in CustomTkinter.

    Args:
        frame: BGR numpy array from OpenCV
        width: Display width
        height: Display height

    Returns:
        CTkImage ready for use in a CTkLabel
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(width, height))


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a numpy embedding array to bytes for SQLite BLOB storage."""
    buffer = io.BytesIO()
    np.save(buffer, embedding)
    return buffer.getvalue()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes from SQLite BLOB back to a numpy embedding array."""
    buffer = io.BytesIO(data)
    return np.load(buffer)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns:
        Float between -1 and 1. Higher = more similar.
        Typically, > 0.4 indicates same person for ArcFace embeddings.
    """
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)


def draw_face_box(
    frame: np.ndarray,
    bbox,
    name: str = "",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box and optional name label on a frame.

    Args:
        frame: BGR image (modified in-place and returned)
        bbox: [x1, y1, x2, y2] coordinates
        name: Optional label to display above the box
        color: BGR color tuple
        thickness: Box line thickness

    Returns:
        The frame with the box drawn on it
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        text_size = cv2.getTextSize(name, font, font_scale, text_thickness)[0]

        # Draw filled rectangle behind text for readability
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0] + 6, y1),
            color,
            -1,
        )
        cv2.putText(frame, name, (x1 + 3, y1 - 5), font, font_scale, (0, 0, 0), text_thickness)

    return frame
