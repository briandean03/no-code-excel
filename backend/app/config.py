from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Excel/CSV Sheet Processor"
    VERSION: str = "1.0.0"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    MIN_DATA_DENSITY: float = 0.3
    MIN_COLS: int = 2
    MIN_ROWS: int = 3
    MAX_GAP: int = 3

settings = Settings()
