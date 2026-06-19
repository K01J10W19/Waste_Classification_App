"""Image preprocessing pipeline for inference — resize, normalize, batch."""
import io
import numpy as np
from PIL import Image
from app.config import settings


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to a normalized (1, H, W, 3) numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((settings.image_size, settings.image_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
