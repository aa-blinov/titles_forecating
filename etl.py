"""
ETL: load raw CSVs, clean, deduplicate, filter opinions, save clean CSVs.
Schema: id, outlet, url, published_at, title, lead, rubric, language, country
"""
import csv
import hashlib
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    RAW_DIR, CLEAN_DIR, OUTLET_SLUGS, SCHEMA_COLS,
    OUTLETS, DEDUP_THRESHOLD,
)

# ================================================================
#  OPINION / BLOG URL PATTERNS TO FILTER OUT
# ================================================================
OPINION_URL_PATTERNS = re.compile(
    r"/(opinion|opinions|blog|blogs|author|authors|column|columns|"
    r"expert|experts|comment|comments|interview|reviews|review|"
    r"мнения|блог|авторы|колонка|рецензии)/",
    re.IGNORECASE,
)
OPINION_RUBRICS = {
    "мнение", "мнения", "блог", "колонка", "авторская колонка",
    "opinion", "blog", "column", "интервью", "рецензия", "рецензии",
}

# ================================================================
#  TEXT CLEANING
# ================================================================

def clean_text(text: Optional[str]) -> Optional[str]:
    """Strip HTML tags, normalize whitespace and unicode."""
    if not text:
        return None
    # Strip residual HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove zero-width and control chars
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    return text if text else None


def normalize_date(val) -> Optional[pd.Timestamp]:
    """Parse date/datetime to UTC-naive pandas Timestamp."""
    if pd.isna(val) if not isinstance(val, str) else not val:
        return None
    try:
        ts = pd.to_datetime(val, utc=False, errors="coerce")
        if ts is pd.NaT:
            return None
        # Drop tz info for uniformity
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


# ================================================================
#  LOAD RAW
# ================================================================

def load_raw(slug: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{slug}_raw.csv")
    if not os.path.exists(path):
        print(f"  [etl] No raw file for {slug}: {path}")
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Ensure schema columns exist
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA_COLS]


# ================================================================
#  CLEAN PIPELINE
# ================================================================

def clean(df: pd.DataFrame, slug: str) -> pd.DataFrame:
    """Full clean pipeline for one outlet's dataframe."""
    print(f"  [etl] {slug}: {len(df)} raw records")

    # 1. Drop empty titles
    df = df[df["title"].str.strip().astype(bool)].copy()

    # 2. Clean text fields
    for col in ["title", "lead", "rubric"]:
        df[col] = df[col].apply(clean_text)

    # 3. Normalize dates
    df["published_at"] = df["published_at"].apply(normalize_date)
    df = df.dropna(subset=["published_at"])

    # 4. Ensure metadata fields
    df["language"] = df["language"].fillna(OUTLETS[slug]["language"])
    df["country"]  = df["country"].fillna(OUTLETS[slug]["country"])
    df["outlet"]   = slug

    # 5. Regenerate IDs from URL for consistency
    df["id"] = df["url"].apply(lambda u: hashlib.md5(str(u).encode()).hexdigest()[:12])

    # 6. Drop rows with identical URLs
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"  [etl] exact URL dedup: {before} -> {len(df)}")

    # 7. Filter opinion / blog content
    df = filter_opinions(df)

    # 8. Near-dedup on title
    df = near_dedup(df)

    df = df.sort_values("published_at").reset_index(drop=True)
    print(f"  [etl] {slug}: {len(df)} clean records")
    return df


def filter_opinions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose URL or rubric indicates opinion/blog content."""
    url_mask    = df["url"].str.contains(OPINION_URL_PATTERNS, na=False)
    rubric_mask = df["rubric"].str.lower().isin(OPINION_RUBRICS).fillna(False)
    removed = url_mask | rubric_mask
    if removed.any():
        print(f"  [etl] opinion filter removed {removed.sum()} rows")
    return df[~removed].copy()


def near_dedup(df: pd.DataFrame, threshold: float = DEDUP_THRESHOLD) -> pd.DataFrame:
    """Remove near-duplicate titles using TF-IDF cosine similarity."""
    titles = df["title"].fillna("").tolist()
    if len(titles) < 2:
        return df

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
    try:
        tfidf = vec.fit_transform(titles)
    except ValueError:
        return df

    keep = [True] * len(titles)
    # Compare each pair sequentially; mark later duplicates
    for i in range(len(titles)):
        if not keep[i]:
            continue
        if i + 1 >= len(titles):
            break
        sims = cosine_similarity(tfidf[i], tfidf[i+1:])[0]
        for j, sim in enumerate(sims, start=i+1):
            if sim >= threshold:
                keep[j] = False

    before = len(df)
    df = df[keep].copy()
    removed = before - len(df)
    if removed:
        print(f"  [etl] near-dedup removed {removed} rows (threshold={threshold})")
    return df


# ================================================================
#  SAVE CLEAN
# ================================================================

def save_clean(slug: str, df: pd.DataFrame) -> str:
    path = os.path.join(CLEAN_DIR, f"{slug}_clean.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ================================================================
#  LOAD CLEAN
# ================================================================

def load_clean(slug: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"{slug}_clean.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


def load_all_clean() -> pd.DataFrame:
    """Concatenate clean datasets for all outlets."""
    frames = [load_clean(slug) for slug in OUTLET_SLUGS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.concat(frames, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df.dropna(subset=["published_at"]).sort_values("published_at")


# ================================================================
#  MAIN ETL ENTRY POINT
# ================================================================

def run_etl(slugs: List[str] = None) -> None:
    if slugs is None:
        slugs = OUTLET_SLUGS
    for slug in slugs:
        df_raw   = load_raw(slug)
        df_clean = clean(df_raw, slug)
        path     = save_clean(slug, df_clean)
        print(f"  [etl] saved {path}")
