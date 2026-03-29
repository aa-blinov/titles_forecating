"""
ETL: load raw CSVs, clean, deduplicate, filter opinions, save clean CSVs.
Schema: id, outlet, url, published_at, title, lead, rubric, language, country
"""
import datetime
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    RAW_DIR, CLEAN_DIR, OUTLET_SLUGS, SCHEMA_COLS,
    OUTLETS, DEDUP_REQUIRE_SAME_RUBRIC, DEDUP_THRESHOLD, DEDUP_WINDOW_HOURS,
)

# ================================================================
#  OPINION / BLOG URL PATTERNS TO FILTER OUT
# ================================================================
OPINION_URL_PATTERNS = re.compile(
    r"/(?:opinion|opinions|blog|blogs|author|authors|column|columns|"
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
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return None
    text = str(text)
    if not text:
        return None
    # Strip residual HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove zero-width and control chars
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    return text if text else None


def clean_title(text: Optional[str], slug: str, rubric: Optional[str] = None) -> Optional[str]:
    """Normalize outlet-specific title artifacts that leak from listing pages."""
    title = clean_text(text)
    if not title:
        return None

    rubric_text = clean_text(rubric) or ""

    if slug == "lenta":
        # Remove listing-page tails like:
        # "...01:17, 18 февраля 2026Силовые структуры"
        if rubric_text:
            rubric_re = re.escape(rubric_text).replace(r"\ ", r"\s*")
            title = re.sub(
                rf"\s*\d{{1,2}}:\d{{2}},\s*\d{{1,2}}\s+[а-яё]+\s+\d{{4}}\s*{rubric_re}\s*$",
                "",
                title,
                flags=re.IGNORECASE,
            )
        else:
            title = re.sub(
                r"\s*\d{1,2}:\d{2},\s*\d{1,2}\s+[а-яё]+\s+\d{4}\s*[А-ЯЁA-Z][^!?]*$",
                "",
                title,
                flags=re.IGNORECASE,
            )

    if slug == "interfax":
        # Remove numeric prefixes glued to digest/photochronicle titles.
        title = re.sub(r"^\d+(?=(?:Фотохроника|Что произошло за день))", "", title)

    title = re.sub(r"\s+", " ", title).strip(" -–|,;:")
    return title or None


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


def _opinion_mask(df: pd.DataFrame) -> pd.Series:
    url_mask = df["url"].str.contains(OPINION_URL_PATTERNS, na=False)
    rubric_mask = df["rubric"].fillna("").str.strip().str.lower().isin(OPINION_RUBRICS)
    return url_mask | rubric_mask


def _audit_step_snapshot(step: str, df: pd.DataFrame,
                         prev_rows: Optional[int] = None,
                         extra: Optional[Dict] = None) -> Dict:
    """Summarize dataset size and date coverage after one ETL step."""
    dates = pd.to_datetime(df["published_at"], errors="coerce") if "published_at" in df.columns else pd.Series(dtype="datetime64[ns]")
    valid_dates = dates.dropna()

    snapshot = {
        "step": step,
        "rows": int(len(df)),
        "removed_since_previous": None if prev_rows is None else int(prev_rows - len(df)),
        "unique_urls": int(df["url"].nunique()) if "url" in df.columns else int(len(df)),
        "duplicate_urls": int(df["url"].duplicated().sum()) if "url" in df.columns else 0,
        "unique_days": int(valid_dates.dt.date.nunique()) if not valid_dates.empty else 0,
        "date_min": valid_dates.min().isoformat() if not valid_dates.empty else None,
        "date_max": valid_dates.max().isoformat() if not valid_dates.empty else None,
        "empty_titles": int(df["title"].fillna("").str.strip().eq("").sum()) if "title" in df.columns else 0,
        "empty_rubrics": int(df["rubric"].fillna("").str.strip().eq("").sum()) if "rubric" in df.columns else 0,
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def _audit_daily_rows(step: str, df: pd.DataFrame) -> List[Dict]:
    """Expand one step into per-day counts for CSV audit output."""
    if "published_at" not in df.columns:
        return []

    dates = pd.to_datetime(df["published_at"], errors="coerce").dropna()
    if dates.empty:
        return []

    counts = dates.dt.date.value_counts().sort_index()
    return [
        {"step": step, "date": day.isoformat(), "count": int(count)}
        for day, count in counts.items()
    ]


# ================================================================
#  CLEAN PIPELINE
# ================================================================

def clean(df: pd.DataFrame, slug: str) -> Tuple[pd.DataFrame, Dict]:
    """Full clean pipeline for one outlet's dataframe."""
    print(f"  [etl] {slug}: {len(df)} raw records")
    audit_steps: List[Dict] = []
    audit_daily: List[Dict] = []

    def record_step(step: str, frame: pd.DataFrame, extra: Optional[Dict] = None) -> None:
        prev_rows = audit_steps[-1]["rows"] if audit_steps else None
        audit_steps.append(_audit_step_snapshot(step, frame, prev_rows=prev_rows, extra=extra))
        audit_daily.extend(_audit_daily_rows(step, frame))

    record_step("raw_input", df)

    # 1. Drop empty titles
    df = df[df["title"].str.strip().astype(bool)].copy()
    record_step("drop_empty_titles", df)

    # 2. Clean text fields
    for col in ["title", "lead", "rubric"]:
        df[col] = df[col].apply(clean_text)

    # 2.1 Clean outlet-specific title artifacts after basic text normalization
    df["title"] = [
        clean_title(title, slug=slug, rubric=rubric)
        for title, rubric in zip(df["title"], df["rubric"])
    ]
    df = df[df["title"].fillna("").str.strip().astype(bool)].copy()
    record_step("clean_titles", df)

    # 3. Normalize dates
    df["published_at"] = df["published_at"].apply(normalize_date)
    df = df.dropna(subset=["published_at"])
    record_step("normalize_dates", df)

    # 4. Ensure metadata fields
    df["language"] = df["language"].replace("", np.nan).fillna(OUTLETS[slug]["language"])
    df["country"]  = df["country"].replace("", np.nan).fillna(OUTLETS[slug]["country"])
    df["outlet"]   = slug

    # 5. Regenerate IDs from URL for consistency
    df["id"] = df["url"].apply(lambda u: hashlib.md5(str(u).encode()).hexdigest()[:12])

    # 6. Drop rows with identical URLs
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"  [etl] exact URL dedup: {before} -> {len(df)}")
    record_step("dedup_exact_url", df)

    # 7. Filter opinion / blog content
    opinion_removed = int(_opinion_mask(df).sum())
    df = filter_opinions(df)
    record_step("filter_opinions", df, extra={"opinion_rows_removed": opinion_removed})

    # 8. Near-dedup on title within a local time window
    df, near_dedup_stats = near_dedup(df)
    record_step("near_dedup", df, extra=near_dedup_stats)

    df = df.sort_values("published_at").reset_index(drop=True)
    record_step("final_clean", df)
    print(f"  [etl] {slug}: {len(df)} clean records")

    audit = {
        "outlet": slug,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dedup_threshold": DEDUP_THRESHOLD,
            "dedup_window_hours": DEDUP_WINDOW_HOURS,
            "dedup_require_same_rubric": DEDUP_REQUIRE_SAME_RUBRIC,
        },
        "steps": audit_steps,
        "daily_counts": audit_daily,
    }
    return df, audit


def filter_opinions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose URL or rubric indicates opinion/blog content."""
    removed = _opinion_mask(df)
    if removed.any():
        print(f"  [etl] opinion filter removed {removed.sum()} rows")
    return df[~removed].copy()


def near_dedup(
    df: pd.DataFrame,
    threshold: float = DEDUP_THRESHOLD,
    window_hours: int = DEDUP_WINDOW_HOURS,
    require_same_rubric: bool = DEDUP_REQUIRE_SAME_RUBRIC,
) -> Tuple[pd.DataFrame, Dict]:
    """Remove near-duplicate titles using local TF-IDF cosine similarity."""
    stats = {
        "near_dedup_removed": 0,
        "near_dedup_threshold": threshold,
        "near_dedup_window_hours": window_hours,
        "near_dedup_require_same_rubric": require_same_rubric,
        "near_dedup_comparisons": 0,
        "near_dedup_examples": [],
    }
    titles = df["title"].fillna("").tolist()
    if len(titles) < 2:
        return df, stats

    df = df.sort_values("published_at").reset_index(drop=True).copy()
    titles = df["title"].fillna("").tolist()
    published = pd.to_datetime(df["published_at"], errors="coerce")
    if published.isna().all():
        return df, stats

    published_ns = published.astype("int64").to_numpy()
    rubrics = df["rubric"].fillna("").astype(str).str.strip().str.lower()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
    try:
        tfidf = vec.fit_transform(titles)
    except ValueError:
        return df, stats

    keep = np.ones(len(titles), dtype=bool)
    window_ns = int(pd.Timedelta(hours=window_hours).value)

    if require_same_rubric:
        groups = [
            idx.to_numpy(dtype=int)
            for _, idx in df.groupby(rubrics, sort=False).groups.items()
        ]
    else:
        groups = [np.arange(len(df), dtype=int)]

    for group_idx in groups:
        if len(group_idx) < 2:
            continue

        group_ts = published_ns[group_idx]
        end_pos = np.searchsorted(group_ts, group_ts + window_ns, side="right")

        for pos_i in range(len(group_idx) - 1):
            i = group_idx[pos_i]
            if not keep[i]:
                continue

            pos_j_end = end_pos[pos_i]
            if pos_j_end <= pos_i + 1:
                continue

            candidate_idx = group_idx[pos_i + 1:pos_j_end]
            candidate_idx = candidate_idx[keep[candidate_idx]]
            if candidate_idx.size == 0:
                continue

            sims = cosine_similarity(tfidf[i], tfidf[candidate_idx])[0]
            stats["near_dedup_comparisons"] += int(candidate_idx.size)
            dup_mask = sims >= threshold
            if not dup_mask.any():
                continue

            dup_idx = candidate_idx[dup_mask]
            keep[dup_idx] = False

            if len(stats["near_dedup_examples"]) < 10:
                sample_idx = dup_idx[: 10 - len(stats["near_dedup_examples"])]
                sample_sims = sims[dup_mask][: len(sample_idx)]
                for j, sim in zip(sample_idx, sample_sims):
                    stats["near_dedup_examples"].append({
                        "kept_date": published.iloc[i].isoformat() if not pd.isna(published.iloc[i]) else None,
                        "removed_date": published.iloc[j].isoformat() if not pd.isna(published.iloc[j]) else None,
                        "similarity": round(float(sim), 4),
                        "kept_rubric": df.iloc[i]["rubric"],
                        "removed_rubric": df.iloc[j]["rubric"],
                        "kept_title": df.iloc[i]["title"],
                        "removed_title": df.iloc[j]["title"],
                    })

    before = len(df)
    df = df[keep].copy()
    removed = before - len(df)
    stats["near_dedup_removed"] = int(removed)
    if removed:
        print(
            f"  [etl] near-dedup removed {removed} rows "
            f"(threshold={threshold}, window={window_hours}h)"
        )
    return df, stats


# ================================================================
#  SAVE CLEAN
# ================================================================

def save_clean(slug: str, df: pd.DataFrame) -> str:
    path = os.path.join(CLEAN_DIR, f"{slug}_clean.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_audit(slug: str, audit: Dict) -> Tuple[str, str]:
    json_path = os.path.join(CLEAN_DIR, f"{slug}_etl_audit.json")
    csv_path = os.path.join(CLEAN_DIR, f"{slug}_etl_daily_counts.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    pd.DataFrame(audit["daily_counts"]).to_csv(csv_path, index=False, encoding="utf-8")
    return json_path, csv_path


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
        df_clean, audit = clean(df_raw, slug)
        path = save_clean(slug, df_clean)
        audit_json_path, audit_csv_path = save_audit(slug, audit)
        print(f"  [etl] saved {path}")
        print(f"  [etl] audit json → {audit_json_path}")
        print(f"  [etl] audit csv  → {audit_csv_path}")
