"""
Helper functions for image conversion, embedding storage, and drawing.
"""

import io
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk


def frame_to_ctkimage(frame, width, height):
    """Convert an OpenCV BGR frame to CTkImage for display."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(width, height))


def embedding_to_bytes(embedding):
    """Pack a numpy array into bytes so we can store it in SQLite."""
    buf = io.BytesIO()
    np.save(buf, embedding)
    return buf.getvalue()


def bytes_to_embedding(data):
    """Unpack bytes from SQLite back into a numpy array."""
    buf = io.BytesIO(data)
    return np.load(buf)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return 0.0
    return float(dot / norms)


def draw_face_box(frame, bbox, name="", color=(0, 255, 0), thickness=2):
    """Draw bounding box on frame with optional name label."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(name, font, 0.7, 2)[0]
        # filled rect behind the text so it's readable
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                       (x1 + text_size[0] + 6, y1), color, -1)
        cv2.putText(frame, name, (x1 + 3, y1 - 5), font, 0.7, (0, 0, 0), 2)

    return frame
