"""Unit tests for the /recommendation endpoint — Phase 8."""
from app.services.recommendation_service import get_recommendations


def test_plastic_recommendation():
    result = get_recommendations("Plastic")
    assert result["waste_label"] == "Plastic"
    assert len(result["tips"]) > 0
    assert result["ranked_methods"][0] == "recycling"


def test_unknown_label_fallback():
    result = get_recommendations("Unknown")
    assert "Please consult your local waste authority." in result["tips"]
