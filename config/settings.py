import os
from dataclasses import dataclass
from dotenv import dotenv_values

_DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


@dataclass
class Config:
    deepseek_api_key: str
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
    flash_model: str
    pro_model: str
    outputs_dir: str
    logs_dir: str

    @classmethod
    def load(cls) -> "Config":
        file = dotenv_values(_DOTENV_PATH)

        def _get(key: str, default: str = "") -> str:
            return file.get(key) or os.getenv(key, default) or default

        def _require(key: str) -> str:
            val = _get(key)
            if not val:
                raise ValueError(f"Missing required env var: {key}")
            return val

        tickers_raw = _get("TICKERS")
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
        return cls(
            deepseek_api_key=_require("DEEPSEEK_API_KEY"),
            smtp_host=_get("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(_get("SMTP_PORT", "465")),
            smtp_user=_require("SMTP_USER"),
            smtp_password=_require("SMTP_PASSWORD"),
            recipient_email=_get("RECIPIENT_EMAIL") or _require("SMTP_USER"),
            tickers=tickers,
            news_lookback_days=int(_get("NEWS_LOOKBACK_DAYS", "1")),
            price_lookback_days=int(_get("PRICE_LOOKBACK_DAYS", "500")),
            training_window_days=int(_get("TRAINING_WINDOW_DAYS", "252")),
            schedule_hour=int(_get("SCHEDULE_HOUR", "9")),
            schedule_minute=int(_get("SCHEDULE_MINUTE", "30")),
            schedule_timezone=_get("SCHEDULE_TIMEZONE", "Europe/London"),
            log_level=_get("LOG_LEVEL", "INFO"),
            flash_model=_get("FLASH_MODEL", "deepseek-v4-pro"),
            pro_model=_get("PRO_MODEL", "deepseek-v4-pro"),
            outputs_dir=_get("OUTPUTS_DIR", "outputs"),
            logs_dir=_get("LOGS_DIR", "logs"),
        )
