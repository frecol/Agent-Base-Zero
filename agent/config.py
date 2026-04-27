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
    deepseek_model: str = "deepseek-v4-flash"
    max_tokens: int = 8192

    # Streaming & Thinking (v0.2)
    stream_enabled: bool = True
    thinking_enabled: bool = True

    # Timezone
    default_timezone: str = "Asia/Shanghai"

    # Context management (v0.3)
    max_context_tokens: int = 64000
    compression_threshold: float = 0.95
    head_keep: int = 3
    tail_keep: int = 20

    # Session (v0.3)
    auto_save: bool = True

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        case_sensitive = False


settings = Settings()