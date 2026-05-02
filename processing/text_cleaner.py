import re

from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_news_item(headline: str, summary: str) -> tuple[str, str]:
    return clean_text(headline), clean_text(summary)
