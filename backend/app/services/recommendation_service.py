"""Rule-based disposal recommendation engine."""

DISPOSAL_TIPS: dict[str, list[str]] = {
    "Plastic": [
        "Place in the recycling bin (blue bin).",
        "Rinse containers before recycling.",
        "Reuse bottles and containers where possible.",
        "Avoid single-use plastics.",
        "Reduce plastic consumption overall.",
    ],
    "Paper": [
        "Place clean, dry paper in the recycling bin.",
        "Avoid recycling paper soiled with food or grease.",
        "Opt for digital documents to reduce paper waste.",
        "Use both sides of paper before discarding.",
    ],
    "Glass": [
        "Place in the glass recycling container.",
        "Rinse jars and bottles before recycling.",
        "Never mix broken glass with regular recyclables.",
        "Reuse glass jars for storage.",
    ],
    "Metal": [
        "Rinse cans before placing in the recycling bin.",
        "Crush cans to save space.",
        "Scrap metals can be taken to a local recycling centre.",
        "Avoid burning metal waste.",
    ],
    "Cardboard": [
        "Flatten boxes before recycling.",
        "Keep cardboard dry — wet cardboard is hard to recycle.",
        "Remove tape and staples where possible.",
        "Compost small, clean cardboard pieces.",
    ],
    "Organic": [
        "Compost organic waste at home or use a green bin.",
        "Avoid sending food waste to landfill.",
        "Use composted material as garden fertiliser.",
        "Reduce food waste by planning meals ahead.",
    ],
}

RANKED_METHODS: dict[str, list[str]] = {
    "Plastic":   ["recycling", "incineration", "landfill"],
    "Paper":     ["recycling", "composting", "landfill"],
    "Glass":     ["recycling", "landfill", "incineration"],
    "Metal":     ["recycling", "incineration", "landfill"],
    "Cardboard": ["recycling", "composting", "landfill"],
    "Organic":   ["composting", "incineration", "landfill"],
}


def get_recommendations(waste_label: str) -> dict:
    """Return tips and ranked disposal methods for a given waste label."""
    label = waste_label.strip().title()
    tips = DISPOSAL_TIPS.get(label, ["Please consult your local waste authority."])
    ranked = RANKED_METHODS.get(label, ["recycling", "landfill", "incineration"])
    return {"waste_label": label, "tips": tips, "ranked_methods": ranked}
