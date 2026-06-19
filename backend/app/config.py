"""Application configuration — loads all settings from environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    climatiq_api_key: str = ""
    model_path: str = "models/waste_classifier.h5"
    image_size: int = 256
    app_env: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
