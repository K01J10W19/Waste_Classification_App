"""GET /recommendation — returns ranked disposal advice for a given waste label."""
from fastapi import APIRouter, HTTPException
from app.schemas.responses import RecommendationResponse
from app.services.recommendation_service import get_recommendations

router = APIRouter()


@router.get("/recommendation", response_model=RecommendationResponse)
async def recommendation(waste_label: str):
    """Return rule-based disposal recommendations ranked by carbon impact."""
    try:
        return get_recommendations(waste_label)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
