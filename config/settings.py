import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Config:
    anthropic_api_key: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    recipient_email: str
    tickers: list[str]
    news_lookback_days: int
    price_lookback_days: int
    training_window_days: int
    schedule_hour: int
    schedule_minute: int
    schedule_timezone: str
    log_level: str
    haiku_model: str
    sonnet_model: str
    outputs_dir: str
    logs_dir: str

    @classmethod
    def load(cls) -> "Config":
        tickers_raw = _optional("TICKERS", "META,KGC,ORCL,IITU,MU")
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
        return cls(
            anthropic_api_key=_require("ANTHROPIC_API_KEY"),
            smtp_host=_optional("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(_optional("SMTP_PORT", "465")),
            smtp_user=_require("SMTP_USER"),
            smtp_password=_require("SMTP_PASSWORD"),
            recipient_email=_optional("RECIPIENT_EMAIL", _require("SMTP_USER")),
            tickers=tickers,
            news_lookback_days=int(_optional("NEWS_LOOKBACK_DAYS", "1")),
            price_lookback_days=int(_optional("PRICE_LOOKBACK_DAYS", "90")),
            training_window_days=int(_optional("TRAINING_WINDOW_DAYS", "60")),
            schedule_hour=int(_optional("SCHEDULE_HOUR", "9")),
            schedule_minute=int(_optional("SCHEDULE_MINUTE", "30")),
            schedule_timezone=_optional("SCHEDULE_TIMEZONE", "Europe/London"),
            log_level=_optional("LOG_LEVEL", "INFO"),
            haiku_model=_optional("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
            sonnet_model=_optional("SONNET_MODEL", "claude-sonnet-4-6"),
            outputs_dir=_optional("OUTPUTS_DIR", "outputs"),
            logs_dir=_optional("LOGS_DIR", "logs"),
        )
