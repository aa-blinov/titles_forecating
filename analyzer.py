"""
Analyzer: topic modeling (TF-IDF + KMeans), frequency analysis,
entity extraction, noise check, style profile.
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    OUTLET_SLUGS, N_TOPICS, TOPIC_WINDOW, CLEAN_DIR, FORECASTS_DIR,
)
from etl import load_clean, load_all_clean

# ================================================================
#  RUSSIAN STOP-WORDS (basic set; NLTK adds more at runtime)
# ================================================================
RU_STOPWORDS_BASE = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об",
    "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "это", "та", "те", "эта", "то", "по", "за", "об",
    # News-specific noise
    "сообщает", "сообщил", "сообщила", "заявил", "заявила",
    "рассказал", "рассказала", "отметил", "отметила",
    "пишет", "пишут", "передает", "передают",
}


def _get_stopwords() -> set:
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return RU_STOPWORDS_BASE | set(stopwords.words("russian"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return RU_STOPWORDS_BASE | set(stopwords.words("russian"))
    except Exception:
        return RU_STOPWORDS_BASE


# ================================================================
#  HELPERS
# ================================================================

def _texts_for_outlet(df: pd.DataFrame) -> List[str]:
    """Combine title and lead into one text per article."""
    texts = []
    for _, row in df.iterrows():
        parts = [str(row.get("title") or "")]
        lead  = row.get("lead")
        if lead and str(lead).lower() not in ("", "nan", "none"):
            parts.append(str(lead))
        texts.append(" ".join(parts))
    return texts


def _clean_token(t: str) -> str:
    return re.sub(r"[^\wа-яА-ЯёЁa-zA-Z0-9]", "", t).lower()


# ================================================================
#  TOPIC MODELING
# ================================================================

def extract_topics(df: pd.DataFrame, outlet: str,
                   n_topics: int = N_TOPICS) -> Tuple[pd.Series, List[str], TfidfVectorizer]:
    """
    TF-IDF + KMeans topic clustering.
    Returns: (cluster_labels Series, cluster_names list, fitted vectorizer)
    """
    texts = _texts_for_outlet(df)
    if len(texts) < n_topics:
        n_topics = max(2, len(texts) // 3)

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=8000,
        stop_words=list(stopwords),
    )
    try:
        tfidf = vec.fit_transform(texts)
    except ValueError:
        return pd.Series([0] * len(df)), ["general"], vec

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(tfidf)

    # Name clusters by top-5 TF-IDF words
    terms = vec.get_feature_names_out()
    cluster_names = []
    for center in km.cluster_centers_:
        top_idx  = center.argsort()[-5:][::-1]
        name     = " / ".join(terms[i] for i in top_idx)
        cluster_names.append(name)

    return pd.Series(labels, index=df.index), cluster_names, vec


def topic_frequency(df: pd.DataFrame, labels: pd.Series,
                    cluster_names: List[str],
                    window: int = TOPIC_WINDOW) -> pd.DataFrame:
    """
    Rolling window topic frequency.
    Returns DataFrame with columns: cluster_id, cluster_name, count, pct
    """
    df2 = df.copy()
    df2["cluster"] = labels

    cutoff = df2["published_at"].max() - pd.Timedelta(days=window)
    recent = df2[df2["published_at"] >= cutoff]

    freq = (
        recent.groupby("cluster").size()
              .reset_index(name="count")
              .sort_values("count", ascending=False)
    )
    freq["cluster_name"] = freq["cluster"].map(
        lambda i: cluster_names[i] if i < len(cluster_names) else f"cluster_{i}"
    )
    total = freq["count"].sum()
    freq["pct"] = (freq["count"] / total * 100).round(1) if total else 0
    return freq


# ================================================================
#  NOISE CHECK
# ================================================================

def noise_check(df: pd.DataFrame, outlet: str) -> Dict:
    """
    Check outlet for:
    - Rubric repetitiveness (top rubric share)
    - Volume stability (daily std/mean ratio)
    """
    result: Dict = {"outlet": outlet}

    # Volume per day
    daily = (
        df.groupby(df["published_at"].dt.date)
          .size()
          .reset_index(name="count")
    )
    if len(daily) > 1:
        mean_vol = daily["count"].mean()
        std_vol  = daily["count"].std()
        cv       = std_vol / mean_vol if mean_vol > 0 else 999
        result["daily_mean"]  = round(mean_vol, 1)
        result["daily_std"]   = round(std_vol, 1)
        result["cv"]          = round(cv, 3)
        result["stable"]      = cv < 0.5
    else:
        result["stable"] = False

    # Top rubric share
    rubric_counts = df["rubric"].value_counts(dropna=True)
    if len(rubric_counts):
        top_rubric  = rubric_counts.index[0]
        top_share   = rubric_counts.iloc[0] / len(df)
        result["top_rubric"]       = top_rubric
        result["top_rubric_share"] = round(top_share, 3)
        result["rubric_noisy"]     = top_share > 0.4
    else:
        result["top_rubric"] = None

    print(f"  [noise] {outlet}: daily_mean={result.get('daily_mean', '?')}, "
          f"cv={result.get('cv', '?')}, stable={result.get('stable', '?')}")
    return result


# ================================================================
#  ENTITY EXTRACTION (lightweight, no mystem required at import)
# ================================================================
_PERSON_TITLES = re.compile(
    r"\b(президент|министр|глава|директор|председатель|генеральный|"
    r"мэр|губернатор|депутат|сенатор|полковник|генерал|адмирал)\b",
    re.IGNORECASE,
)
_CAPS_TOKEN = re.compile(r"^[А-ЯЁA-Z][а-яёa-z]{2,}$")


def extract_entities(texts: List[str]) -> Dict[str, List[str]]:
    """
    Lightweight entity extraction using capitalization heuristics
    and pymystem3 (if available).
    Returns dict with keys: persons, orgs, locations (lists of strings).
    """
    persons: List[str]   = []
    orgs: List[str]      = []
    locations: List[str] = []

    # Try pymystem3 for better accuracy
    try:
        from pymystem3 import Mystem
        mystem = Mystem()
        combined = " ".join(texts[:500])  # limit to avoid timeout
        analysis = mystem.analyze(combined)
        for token_info in analysis:
            word = token_info.get("text", "").strip()
            analysis_list = token_info.get("analysis", [])
            if not analysis_list:
                continue
            gr = analysis_list[0].get("gr", "")
            if not _CAPS_TOKEN.match(word):
                continue
            if "Persn" in gr or "Name" in gr:
                persons.append(word)
            elif "Patr" in gr or "Famn" in gr:
                persons.append(word)
            elif "Orgn" in gr:
                orgs.append(word)
            elif "Geox" in gr or "Surn" in gr:
                locations.append(word)

    except Exception:
        # Fallback: capitalized word bigrams after person titles
        for text in texts:
            tokens = text.split()
            for i, tok in enumerate(tokens):
                if _CAPS_TOKEN.match(tok) and i > 0:
                    if _PERSON_TITLES.search(tokens[i - 1]):
                        persons.append(tok)
                    elif tok.isupper() and len(tok) >= 3:
                        orgs.append(tok)

    return {
        "persons":   persons,
        "orgs":      orgs,
        "locations": locations,
    }


def entity_frequency(df: pd.DataFrame, top_n: int = 20,
                     window_days: int = 14) -> Dict[str, pd.Series]:
    """Top entities in recent N days."""
    cutoff = df["published_at"].max() - pd.Timedelta(days=window_days)
    recent = df[df["published_at"] >= cutoff]
    texts  = _texts_for_outlet(recent)
    ents   = extract_entities(texts)

    result = {}
    for key, lst in ents.items():
        if lst:
            s = pd.Series(lst).value_counts().head(top_n)
            result[key] = s
        else:
            result[key] = pd.Series(dtype=int)
    return result


# ================================================================
#  STYLE PROFILE
# ================================================================

def style_profile(df: pd.DataFrame) -> Dict:
    """
    Compute basic style metrics for an outlet corpus:
    avg title length (words), avg lead length (words), top bigrams.
    """
    titles  = df["title"].dropna()
    leads   = df["lead"].dropna()

    title_lens = titles.apply(lambda t: len(str(t).split()))
    lead_lens  = leads.apply(lambda t: len(str(t).split()))

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        ngram_range=(2, 2), max_features=30,
        stop_words=list(stopwords),
    )
    if len(titles) >= 5:
        try:
            vec.fit(titles)
            bigrams = list(vec.get_feature_names_out())
        except Exception:
            bigrams = []
    else:
        bigrams = []

    return {
        "avg_title_len":  round(title_lens.mean(), 1) if len(title_lens) else 0,
        "avg_lead_len":   round(lead_lens.mean(), 1)  if len(lead_lens) else 0,
        "top_bigrams":    bigrams[:10],
    }


# ================================================================
#  FULL ANALYZE PIPELINE
# ================================================================

def analyze_outlet(slug: str) -> Dict:
    df = load_clean(slug)
    if df.empty:
        print(f"  [analyze] No clean data for {slug}")
        return {}

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    print(f"\n[analyze] {slug}: {len(df)} records")

    labels, names, vec = extract_topics(df, slug)
    freq        = topic_frequency(df, labels, names)
    noise       = noise_check(df, slug)
    ents        = entity_frequency(df)
    style       = style_profile(df)

    print(f"  [topics] top-5:")
    for _, row in freq.head(5).iterrows():
        print(f"    {row['cluster_name']}: {row['count']} ({row['pct']}%)")

    print(f"  [entities] top persons: {ents['persons'].head(5).to_dict()}")
    print(f"  [style]  avg_title={style['avg_title_len']} words, "
          f"avg_lead={style['avg_lead_len']} words")

    return {
        "slug":          slug,
        "df":            df,
        "labels":        labels,
        "cluster_names": names,
        "vectorizer":    vec,
        "topic_freq":    freq,
        "noise":         noise,
        "entities":      ents,
        "style":         style,
    }


def analyze_all(slugs: List[str] = None) -> Dict[str, Dict]:
    if slugs is None:
        slugs = OUTLET_SLUGS
    return {slug: analyze_outlet(slug) for slug in slugs}
