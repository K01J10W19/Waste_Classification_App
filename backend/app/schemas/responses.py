"""Pydantic schemas for all API request and response bodies."""
from pydantic import BaseModel, Field
from typing import List


class ClassificationResponse(BaseModel):
    waste_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: dict[str, float]


class CarbonEstimateRequest(BaseModel):
    waste_label: str
    weight_kg: float = Field(..., gt=0)
    location: str


class DisposalPathEstimate(BaseModel):
    method: str          # "recycling" | "landfill" | "incineration"
    co2e_kg: float
    unit: str = "kg CO2e"


class CarbonEstimateResponse(BaseModel):
    waste_label: str
    weight_kg: float
    location: str
    estimates: List[DisposalPathEstimate]
    recommended_method: str   # lowest-emission path


class RecommendationResponse(BaseModel):
    waste_label: str
    tips: List[str]
    ranked_methods: List[str]
