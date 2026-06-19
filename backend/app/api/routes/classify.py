"""POST /classify — accepts a waste image and returns the predicted label + confidence."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.responses import ClassificationResponse
from app.ml.predict import predict_waste

router = APIRouter()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_waste(file: UploadFile = File(...)):
    """Run CNN inference on an uploaded waste image."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG/WebP images are accepted.")
    try:
        image_bytes = await file.read()
        result = predict_waste(image_bytes)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
