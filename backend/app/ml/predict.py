"""Runs CNN inference and returns the predicted waste label and confidence."""
import numpy as np
from app.ml.model_loader import get_model
from app.ml.preprocess import preprocess_image

WASTE_CLASSES = ["Cardboard", "Glass", "Metal", "Organic", "Paper", "Plastic"]


def predict_waste(image_bytes: bytes) -> dict:
    """Return predicted label, confidence, and full score map."""
    model = get_model()
    tensor = preprocess_image(image_bytes)
    scores = model.predict(tensor)[0]
    idx = int(np.argmax(scores))
    return {
        "waste_label": WASTE_CLASSES[idx],
        "confidence": float(scores[idx]),
        "all_scores": {cls: float(scores[i]) for i, cls in enumerate(WASTE_CLASSES)},
    }
