"""Configuration management.

Reads settings from environment variables or a .env file.
Uses pydantic-settings for type-safe, validated configuration.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek API
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    max_tokens: int = 4096

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        case_sensitive = False


settings = Settings()
