"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import classify, carbon, recommend

app = FastAPI(
    title="Waste Classification API",
    description="CNN-based waste classification with carbon impact estimation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api")
app.include_router(carbon.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
