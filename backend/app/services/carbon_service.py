"""Climatiq API client — fetches CO2e estimates for multiple disposal paths."""
import httpx
from app.config import settings

CLIMATIQ_URL = "https://beta3.api.climatiq.io/estimate"

# Maps CNN output label → Climatiq activity IDs per disposal method.
# These IDs will be refined once Climatiq activity library is confirmed.
LABEL_ACTIVITY_MAP = {
    "Plastic":   {"recycling": "waste_management-waste_type_plastic-disposal_method_recycling",
                  "landfill":  "waste_management-waste_type_plastic-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_plastic-disposal_method_incineration"},
    "Paper":     {"recycling": "waste_management-waste_type_paper-disposal_method_recycling",
                  "landfill":  "waste_management-waste_type_paper-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_paper-disposal_method_incineration"},
    "Glass":     {"recycling": "waste_management-waste_type_glass-disposal_method_recycling",
                  "landfill":  "waste_management-waste_type_glass-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_glass-disposal_method_incineration"},
    "Metal":     {"recycling": "waste_management-waste_type_metal-disposal_method_recycling",
                  "landfill":  "waste_management-waste_type_metal-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_metal-disposal_method_incineration"},
    "Cardboard": {"recycling": "waste_management-waste_type_paper_cardboard-disposal_method_recycling",
                  "landfill":  "waste_management-waste_type_paper_cardboard-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_paper_cardboard-disposal_method_incineration"},
    "Organic":   {"recycling": "waste_management-waste_type_organic-disposal_method_composting",
                  "landfill":  "waste_management-waste_type_organic-disposal_method_landfill",
                  "incineration": "waste_management-waste_type_organic-disposal_method_incineration"},
}


async def get_carbon_estimates(waste_label: str, weight_kg: float, location: str) -> dict:
    """Call Climatiq for recycling/landfill/incineration CO2e values."""
    activities = LABEL_ACTIVITY_MAP.get(waste_label)
    if not activities:
        raise ValueError(f"No Climatiq activity mapping for label: {waste_label}")

    headers = {"Authorization": f"Bearer {settings.climatiq_api_key}"}
    estimates = []

    async with httpx.AsyncClient() as client:
        for method, activity_id in activities.items():
            payload = {
                "emission_factor": {"activity_id": activity_id, "region": location},
                "parameters": {"weight": weight_kg, "weight_unit": "kg"},
            }
            resp = await client.post(CLIMATIQ_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            estimates.append({
                "method": method,
                "co2e_kg": data["co2e"],
                "unit": "kg CO2e",
            })

    recommended = min(estimates, key=lambda x: x["co2e_kg"])["method"]
    return {
        "waste_label": waste_label,
        "weight_kg": weight_kg,
        "location": location,
        "estimates": estimates,
        "recommended_method": recommended,
    }
