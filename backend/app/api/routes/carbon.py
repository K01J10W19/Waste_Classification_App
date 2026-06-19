"""POST /carbon-estimate — returns CO2e estimates for multiple disposal paths."""
from fastapi import APIRouter, HTTPException
from app.schemas.responses import CarbonEstimateRequest, CarbonEstimateResponse
from app.services.carbon_service import get_carbon_estimates

router = APIRouter()


@router.post("/carbon-estimate", response_model=CarbonEstimateResponse)
async def carbon_estimate(payload: CarbonEstimateRequest):
    """Call Climatiq API for recycling/landfill/incineration CO2 estimates."""
    try:
        result = await get_carbon_estimates(
            waste_label=payload.waste_label,
            weight_kg=payload.weight_kg,
            location=payload.location,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
