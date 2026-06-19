"""Shared FastAPI dependencies (e.g., model singleton injection)."""
from app.ml.model_loader import get_model


def model_dependency():
    """Dependency that ensures the model is loaded before a request is handled."""
    return get_model()
