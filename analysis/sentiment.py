import torch
from transformers import pipeline as hf_pipeline
from loguru import logger

from models.types import NewsItem, SentimentResult, ScoredNewsItem, TickerSentiment


_LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_MAX_TOKENS = 512


class SentimentAnalyser:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}")
        self._pipe = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=device,
            truncation=True,
            max_length=_MAX_TOKENS,
        )

    def score_batch(self, items: list[NewsItem]) -> list[ScoredNewsItem]:
        if not items:
            return []

        texts = [
            f"{item.headline}. {item.summary}"[:1024]  # hard cap before tokenization
            for item in items
        ]

        try:
            raw_results = self._pipe(texts, batch_size=16)
        except Exception as exc:
            logger.warning(f"FinBERT batch failed — {exc}; scoring individually")
            raw_results = []
            for text in texts:
                try:
                    raw_results.append(self._pipe(text)[0])
                except Exception:
                    raw_results.append({"label": "neutral", "score": 0.5})

        scored = []
        for item, result in zip(items, raw_results):
            label = result["label"].lower()
            confidence = float(result["score"])
            scored.append(ScoredNewsItem(
                news_item=item,
                sentiment=SentimentResult(label=label, confidence=confidence),
            ))

        logger.debug(f"Scored {len(scored)} news items with FinBERT")
        return scored

    def aggregate_by_ticker(
        self,
        scored_items: list[ScoredNewsItem],
        tickers: list[str],
        date: str,
    ) -> dict[str, TickerSentiment]:
        ticker_map: dict[str, list[ScoredNewsItem]] = {t: [] for t in tickers}

        for scored in scored_items:
            for ticker in scored.news_item.tickers:
                if ticker in ticker_map:
                    ticker_map[ticker].append(scored)

        result: dict[str, TickerSentiment] = {}
        for ticker, items in ticker_map.items():
            if not items:
                result[ticker] = TickerSentiment(
                    ticker=ticker,
                    date=date,
                    avg_score=0.0,
                    article_count=0,
                    positive_count=0,
                    neutral_count=0,
                    negative_count=0,
                    top_headlines=[],
                    top_news_urls=[],
                )
                continue

            scores = [_LABEL_TO_SCORE[i.sentiment.label] for i in items]
            avg_score = sum(scores) / len(scores)
            pos = sum(1 for i in items if i.sentiment.label == "positive")
            neu = sum(1 for i in items if i.sentiment.label == "neutral")
            neg = sum(1 for i in items if i.sentiment.label == "negative")

            # Pick top 3 headlines: positive first, then by confidence
            sorted_items = sorted(
                items,
                key=lambda x: (x.sentiment.label == "positive", x.sentiment.confidence),
                reverse=True,
            )
            top_items = sorted_items[:3]
            top_headlines = [i.news_item.headline for i in top_items]
            top_news_urls = [i.news_item.url for i in top_items]

            result[ticker] = TickerSentiment(
                ticker=ticker,
                date=date,
                avg_score=round(avg_score, 4),
                article_count=len(items),
                positive_count=pos,
                neutral_count=neu,
                negative_count=neg,
                top_headlines=top_headlines,
                top_news_urls=top_news_urls,
            )

        return result
