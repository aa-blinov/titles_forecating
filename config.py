"""
Configuration: outlets, paths, dates, Ollama settings.
"""
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================================================================
#  BASE DIRECTORY — everything lives inside titles/
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
CLEAN_DIR       = os.path.join(DATA_DIR, "clean")
FORECASTS_DIR   = os.path.join(DATA_DIR, "forecasts")
EVENTS_DIR      = os.path.join(DATA_DIR, "events")
EVENTS_CSV      = os.path.join(EVENTS_DIR, "events.csv")

for _d in [RAW_DIR, CLEAN_DIR, FORECASTS_DIR, EVENTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ================================================================
#  DATES
# ================================================================
HISTORY_DAYS = 90
TODAY        = datetime.date.today()
HISTORY_FROM = TODAY - datetime.timedelta(days=HISTORY_DAYS)
TARGET_DATE  = datetime.date(2026, 4, 2)

# ================================================================
#  OUTLETS
# ================================================================
# Keys: short slug used as filename prefix and dict key everywhere.
OUTLETS = {
    "kommersant": {
        "name": "Коммерсантъ",
        "url": "https://www.kommersant.ru/",
        "rss": [
            "https://www.kommersant.ru/RSS/news.xml",
            "https://www.kommersant.ru/RSS/sect-4.xml",   # Business
            "https://www.kommersant.ru/RSS/sect-3.xml",   # Politics
        ],
        "archive_url": "https://www.kommersant.ru/archive/rubric/1/year/{year}/month/{month:02d}",
        "lead_selector": "p.js-search-mark",
        "rubric_selector": "a.article_subheader",
        "has_lead": True,
        "language": "ru",
        "country": "RU",
    },
    "lenta": {
        "name": "Лента.ру",
        "url": "https://lenta.ru/",
        "rss": [
            "https://lenta.ru/rss/news",
            "https://lenta.ru/rss/top7",
        ],
        "archive_url": "https://lenta.ru/news/{year}/{month:02d}/{day:02d}/",
        "lead_selector": "div.topic-body__content p",
        "rubric_selector": "a.topic-header__item",
        "has_lead": True,
        "language": "ru",
        "country": "RU",
    },
    "interfax": {
        "name": "Интерфакс",
        "url": "https://www.interfax.ru/",
        "rss": [
            "https://www.interfax.ru/rss.asp",
        ],
        "archive_url": "https://www.interfax.ru/news/{year}/{month:02d}/{day:02d}",
        "lead_selector": "article.ifx-text p",
        "rubric_selector": "div.rubric a",
        "has_lead": True,
        "language": "ru",
        "country": "RU",
    },
}

OUTLET_SLUGS = list(OUTLETS.keys())

# ================================================================
#  SCRAPER SETTINGS
# ================================================================
REQUEST_DELAY   = 1.5       # seconds between requests
REQUEST_TIMEOUT = 15        # seconds
MAX_RETRIES     = 3
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

# Wayback Machine CDX API (fallback for archive scraping)
WAYBACK_CDX_URL = (
    "http://web.archive.org/cdx/search/cdx"
    "?url={url}*&output=json&fl=timestamp,original&limit=500"
    "&from={from_ts}&to={to_ts}&statuscode=200&matchType=prefix"
)

# ================================================================
#  OPENROUTER SETTINGS
# ================================================================
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL      = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku-20240307")

# ================================================================
#  ANALYSIS SETTINGS
# ================================================================
N_TOPICS        = 20       # KMeans clusters per outlet
TOPIC_WINDOW    = 14       # days for rolling topic frequency
FREQ_WINDOW     = 14       # days for headline frequency baseline
BACKTEST_DAYS   = 7        # holdout window for backtesting

# ================================================================
#  DEDUP THRESHOLD
# ================================================================
DEDUP_THRESHOLD = 0.85     # cosine similarity threshold for near-dedup

# ================================================================
#  SCHEMA
# ================================================================
SCHEMA_COLS = [
    "id", "outlet", "url", "published_at",
    "title", "lead", "rubric", "language", "country"
]
