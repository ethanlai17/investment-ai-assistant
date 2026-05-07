import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import markdown as md_lib
from loguru import logger


class EmailSender:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        recipient_email: str,
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._recipient = recipient_email

    def send(self, md_path: Path, txt_path: Path, report_date: str, subject: str | None = None) -> None:
        subject = subject or f"Investment Report — {report_date}"
        txt_content = txt_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")
        html_content = self._md_to_html(md_content)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._user
        msg["To"] = self._recipient
        msg.attach(MIMEText(txt_content, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        with smtplib.SMTP_SSL(self._host, self._port) as server:
            server.login(self._user, self._password)
            server.sendmail(self._user, self._recipient, msg.as_string())

        logger.info(f"Report emailed to {self._recipient} for {report_date}")

    def _md_to_html(self, md_content: str) -> str:
        body = md_lib.markdown(
            md_content,
            extensions=["tables", "nl2br"],
        )
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 28px; }}
  h3 {{ color: #0f3460; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #f4f4f4; font-weight: bold; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 24px 0; }}
  ul {{ padding-left: 20px; }}
  li {{ margin: 4px 0; }}
  strong {{ color: #1a1a2e; }}
  em {{ color: #666; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
