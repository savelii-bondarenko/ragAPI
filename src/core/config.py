import os
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    QDRANT_PATH: str = os.path.join(BASE_DIR, "qdrant_data")
    QDRANT_COLLECTION: str = "all_documents"
    VDB_SEARCH_K: int = 4

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    LLM_MODEL: str = "deepseek-chat"
    DEEPSEEK_API_KEY: str  

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()