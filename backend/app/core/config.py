from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
UPLOADS_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "lawyer_ai.db"
PROMPT_PATH = BACKEND_DIR / "app" / "core" / "prompts" / "lawyer_ai_system.md"

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    debug: bool = Field(default=False, alias="DEBUG")
    default_state: str = Field(default="Gujarat", alias="DEFAULT_STATE")
    retrieval_mode: str = Field(default="local_context", alias="RETRIEVAL_MODE")
    openai_timeout_seconds: float = Field(default=45.0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=2, alias="OPENAI_MAX_RETRIES")
    database_backend: str = Field(default="sqlite", alias="DATABASE_BACKEND")
    mongodb_uri: str = Field(default="mongodb://127.0.0.1:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="lawyer_ai", alias="MONGODB_DATABASE")
    cors_origins_raw: str = Field(
        default="http://127.0.0.1:5000,http://localhost:5000",
        alias="CORS_ALLOW_ORIGINS",
    )
    openai_vector_store_id: str = Field(default="", alias="OPENAI_VECTOR_STORE_ID")
    google_client_id: str = Field(default="", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="", alias="GOOGLE_REDIRECT_URI")
    auth_cookie_name: str = Field(default="legal_auth_token", alias="AUTH_COOKIE_NAME")
    auth_session_duration_days: int = Field(default=14, alias="AUTH_SESSION_DURATION_DAYS")
    frontend_auth_path: str = Field(default="/frontend/auth.html", alias="FRONTEND_AUTH_PATH")
    frontend_app_path: str = Field(default="/frontend/index.html", alias="FRONTEND_APP_PATH")
    app_base_url: str = Field(default="http://127.0.0.1:5000", alias="APP_BASE_URL")
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_username: str = Field(default="", alias="SMTP_USERNAME")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from_email: str = Field(default="", alias="SMTP_FROM_EMAIL")
    smtp_from_name: str = Field(default="Lawyer AI", alias="SMTP_FROM_NAME")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")
    password_reset_token_ttl_minutes: int = Field(default=30, alias="PASSWORD_RESET_TOKEN_TTL_MINUTES")
    indiankanoon_api_base_url: str = Field(
        default="https://api.indiankanoon.org",
        alias="INDIANKANOON_API_BASE_URL",
    )
    indiankanoon_api_token: str = Field(
        default="",
        alias="INDIANKANOON_API_TOKEN",
        validation_alias=AliasChoices("INDIANKANOON_API_TOKEN", "INDIA_KANOON_API_TOKEN"),
    )
    indiankanoon_timeout_seconds: float = Field(default=20.0, alias="INDIANKANOON_TIMEOUT_SECONDS")
    indiankanoon_trust_env_proxy: bool = Field(default=False, alias="INDIANKANOON_TRUST_ENV_PROXY")
    google_search_enabled: bool = Field(default=False, alias="GOOGLE_SEARCH_ENABLED")
    google_custom_search_api_key: str = Field(default="", alias="GOOGLE_CUSTOM_SEARCH_API_KEY")
    google_custom_search_cx: str = Field(default="", alias="GOOGLE_CUSTOM_SEARCH_CX")
    google_search_timeout_seconds: float = Field(default=8.0, alias="GOOGLE_SEARCH_TIMEOUT_SECONDS")
    google_search_max_results: int = Field(default=3, alias="GOOGLE_SEARCH_MAX_RESULTS")
    google_search_cache_ttl_seconds: int = Field(default=21600, alias="GOOGLE_SEARCH_CACHE_TTL_SECONDS")
    google_search_trusted_domains_raw: str = Field(
        default="gov.in,nic.in,rbi.org.in,cybercrime.gov.in,sci.gov.in",
        alias="GOOGLE_SEARCH_TRUSTED_DOMAINS",
    )
    local_model_assist_enabled: bool = Field(default=True, alias="LOCAL_MODEL_ASSIST_ENABLED")
    local_text_similarity_model: str = Field(default="", alias="LOCAL_TEXT_SIMILARITY_MODEL")
    local_support_check_enabled: bool = Field(default=True, alias="LOCAL_SUPPORT_CHECK_ENABLED")
    local_support_check_threshold: float = Field(default=0.58, alias="LOCAL_SUPPORT_CHECK_THRESHOLD")

    @field_validator("debug", mode="before")
    @classmethod
    def coerce_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production", "warn", "warning", "info", "error"}:
            return False
        return value

    @field_validator("database_backend", mode="before")
    @classmethod
    def normalize_database_backend(cls, value):
        normalized = str(value or "sqlite").strip().lower()
        if normalized in {"sqlite", "mongodb"}:
            return normalized
        raise ValueError("DATABASE_BACKEND must be either 'sqlite' or 'mongodb'")

    @field_validator("indiankanoon_trust_env_proxy", mode="before")
    @classmethod
    def coerce_indiankanoon_trust_env_proxy(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return value

    @field_validator(
        "local_model_assist_enabled",
        "local_support_check_enabled",
        "google_search_enabled",
        mode="before",
    )
    @classmethod
    def coerce_local_bool(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    @property
    def google_search_trusted_domains(self) -> list[str]:
        return [item.strip().lower() for item in self.google_search_trusted_domains_raw.split(",") if item.strip()]

    @property
    def database_url(self) -> str:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return str(DB_PATH)

    @property
    def frontend_dir(self) -> Path:
        if FRONTEND_DIR.exists():
            return FRONTEND_DIR
        alternate = ROOT_DIR / "Frontend"
        return alternate

    @property
    def knowledge_dir(self) -> Path:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        return KNOWLEDGE_DIR

    @property
    def uploads_dir(self) -> Path:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        return UPLOADS_DIR

    @property
    def temp_dir(self) -> Path:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return TEMP_DIR

    @property
    def prompt_path(self) -> Path:
        return PROMPT_PATH

    @property
    def domain_packs_manifest_path(self) -> Path:
        return DATA_DIR / "manifests" / "legal_domain_packs.json"

    @property
    def smtp_configured(self) -> bool:
        required = [
            self.smtp_host.strip(),
            self.smtp_username.strip(),
            self.smtp_password.strip(),
            self.smtp_from_email.strip(),
        ]
        return all(required)

    @property
    def indiankanoon_configured(self) -> bool:
        return bool(self.indiankanoon_api_token.strip())

    @property
    def google_custom_search_configured(self) -> bool:
        return (
            self.google_search_enabled
            and bool(self.google_custom_search_api_key.strip())
            and bool(self.google_custom_search_cx.strip())
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
