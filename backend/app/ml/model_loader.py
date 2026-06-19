"""Loads the trained Keras model once at application startup."""
import tensorflow as tf
from app.config import settings

_model = None


def get_model() -> tf.keras.Model:
    """Return the cached model instance, loading it on first call."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(settings.model_path)
    return _model
