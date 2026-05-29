from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Typed configuration module using pydantic_settings.
    Automatically loads secrets from environment variables or a .env file 
    to avoid hardcoded credentials.
    """
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    random_seed: int = 42
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# Singleton settings instance
settings = Settings()
