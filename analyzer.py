"""
Analyzer: topic modeling (TF-IDF + KMeans), frequency analysis,
entity extraction, noise check, style profile.
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    OUTLET_SLUGS, N_TOPICS, TOPIC_WINDOW, CLEAN_DIR, FORECASTS_DIR,
)
from etl import load_clean

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

RU_STOPWORDS_EXTRA = {
    # Months (various forms)
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    # News fillers & common words
    "года", "году", "это", "свой", "который", "которые",
    "стало", "известно", "сообщили", "ранее", "подробности", "выяснилось",
    "назван", "названа", "названы", "впервые", "появилось", "появились",
    "рассказали", "оказалось", "оказались", "дня", "дней",
}


def _get_stopwords() -> set:
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            base = RU_STOPWORDS_BASE | set(stopwords.words("russian"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            base = RU_STOPWORDS_BASE | set(stopwords.words("russian"))
        return base | RU_STOPWORDS_EXTRA
    except Exception:
        return RU_STOPWORDS_BASE | RU_STOPWORDS_EXTRA


# ================================================================
#  HELPERS
# ================================================================

def _texts_for_outlet(df: pd.DataFrame) -> List[str]:
    """Combine title and lead into one text per article (vectorized)."""
    titles = df["title"].fillna("").astype(str)
    leads  = df["lead"].fillna("").astype(str)
    # Blank out literal "nan" / "none" strings
    leads  = leads.where(~leads.str.lower().isin(["nan", "none", ""]), "")
    combined = (titles + " " + leads).str.strip()
    return combined.tolist()


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
    unique_texts = set(t.strip() for t in texts if t.strip())
    n = len(texts)
    n_unique = len(unique_texts)

    if n_unique < n_topics:
        n_topics = max(1, n_unique - 1)
    if n_topics < 2:
        # Fallback for extremely small datasets
        return pd.Series([0] * len(df), index=df.index), ["general"], None

    # Adaptive min_df: at least 2, but scales with corpus size
    adaptive_min_df = max(2, n // 50)

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[а-яА-ЯёЁa-zA-Z]{3,}\b",
        min_df=adaptive_min_df,
        max_df=0.85,
        max_features=5000,
        stop_words=list(stopwords),
    )
    try:
        tfidf = vec.fit_transform(texts)
    except ValueError:
        return pd.Series([0] * len(df), index=df.index), ["general"], vec

    # Cap n_topics by number of unique rows in TF-IDF matrix to avoid ConvergenceWarning
    if tfidf.shape[0] > 0:
        n_distinct_samples = len(np.unique(tfidf.toarray(), axis=0))
        if n_topics >= n_distinct_samples:
            n_topics = max(1, n_distinct_samples - 1)

    if n_topics < 2:
        return pd.Series([0] * len(df), index=df.index), ["general"], vec

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(tfidf)

    # Name clusters by top-5 TF-IDF words
    terms = vec.get_feature_names_out()
    cluster_names = []
    for center in km.cluster_centers_:
        top_idx = center.argsort()[-5:][::-1]
        name    = " / ".join(terms[i] for i in top_idx)
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

    max_date = df2["published_at"].max()
    min_date = df2["published_at"].min()
    # Don't let window exceed actual data range
    actual_days = max(1, (max_date - min_date).days)
    effective_window = min(window, actual_days)

    cutoff = max_date - pd.Timedelta(days=effective_window)
    recent = df2[df2["published_at"] >= cutoff]

    if recent.empty:
        recent = df2  # fallback: use all data

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
        cv       = std_vol / daily_mean if daily_mean > 0 else 999
    else:
        daily_mean = float(df.shape[0])
        cv = 999.0

    # Rubric noise check
    counts = df["rubric"].str.strip().replace("", np.nan).dropna().value_counts()
    n_missing = (df["rubric"].str.strip() == "").sum()
    
    top_rubric = ""
    top_share  = 0
    if not counts.empty:
        top_rubric = counts.index[0]
        # Noise is the share of missing or the largest single rubric share
        top_share = max(n_missing / len(df), counts.iloc[0] / len(df)) if len(df) > 0 else 0
    else:
        top_share = n_missing / len(df) if len(df) > 0 else 0

    result = {
        "outlet":          outlet,
        "daily_mean":      float(daily_mean),
        "cv":              float(cv),
        "stable":          bool(cv < 0.5),
        "top_rubric":      top_rubric if top_rubric else "EMPTY",
        "top_rubric_share": float(top_share),
        "rubric_noisy":    bool(top_share > 0.6),  # 60%+ share of one value/empty is noisy
    }

    print(f"  [noise] {outlet}: daily_mean={result.get('daily_mean', '?')}, "
          f"cv={result.get('cv', '?')}, stable={result.get('stable', '?')}")
    return result


# ================================================================
#  ENTITY EXTRACTION
#  Pure regex/heuristic — no external NLP (pymystem3 hangs on macOS ARM)
# ================================================================
_PERSON_TITLES = re.compile(
    r"\b(президент|министр|глава|директор|председатель|генеральный|"
    r"мэр|губернатор|депутат|сенатор|полковник|генерал|адмирал|"
    r"спикер|лидер|руководитель|секретарь|посол|прокурор|судья|"
    r"премьер|вице-премьер|вице|командующий|начальник)\b",
    re.IGNORECASE,
)
_ORG_MARKERS = re.compile(
    r"\b(министерство|ведомство|правительство|дума|совет|суд|банк|"
    r"компания|корпорация|агентство|служба|федерация|союз|ассоциация|"
    r"партия|комитет|фонд|институт|университет|академия|холдинг|"
    r"медиагруппа|группа)\b",
    re.IGNORECASE,
)
_GEO_MARKERS = re.compile(
    r"\b(россия|москва|украина|сша|китай|европа|германия|франция|"
    r"великобритания|беларусь|казахстан|азия|африка|ближний|восток|"
    r"петербург|новосибирск|екатеринбург|краснодар|регион|область|"
    r"республика|округ|край|город|страна)\b",
    re.IGNORECASE,
)
# Capitalized word: starts with uppercase, rest lowercase, min 3 chars
_CAPS_TOKEN = re.compile(r"^[А-ЯЁA-Z][а-яёa-z]{2,}$")
# All-caps abbreviation: МВД, ФСБ, ЦБ, США  (2+ chars, no digits)
_CAPS_ABBR  = re.compile(r"^[А-ЯЁA-Z]{2,}$")
# Known non-org abbreviations to skip
_SKIP_ABBR  = {"НДС", "ВВП", "РФ", "ВВС", "СМИ", "ЖКХ", "ЕС", "ООН",
               "МВФ", "ВОЗ", "НКО", "КПД", "ДТП", "ЧП", "ЧС"}


def extract_entities(texts: List[str]) -> Dict[str, List[str]]:
    """
    Lightweight entity extraction using capitalization heuristics.
    Returns dict with keys: persons, orgs, locations (lists of strings).

    Strategy:
    • 2-3 consecutive Capitalized words → person (unless known geo/org)
    • All-caps abbreviations (МВД, ФСБ) → org, with known noise filtered
    • Single Capitalized word near geo/org marker → location / org
    • Single Capitalized word after person-title → person
    """
    persons: List[str]   = []
    orgs: List[str]      = []
    locations: List[str] = []

    for text in texts:
        tokens = text.split()
        n = len(tokens)
        i = 0
        while i < n:
            tok = tokens[i]

            # All-caps abbreviation → likely an org (skip known noise)
            if _CAPS_ABBR.match(tok) and len(tok) >= 2 and tok not in _SKIP_ABBR:
                orgs.append(tok)
                i += 1
                continue

            if not _CAPS_TOKEN.match(tok):
                i += 1
                continue

            # Gather run of consecutive Capitalized tokens
            run = [tok]
            j = i + 1
            while j < n and _CAPS_TOKEN.match(tokens[j]):
                run.append(tokens[j])
                j += 1

            if 2 <= len(run) <= 3:
                # Multi-word Capitalized run → person, unless geo/org
                candidate = " ".join(run)
                if _GEO_MARKERS.search(candidate):
                    locations.append(candidate)
                elif _ORG_MARKERS.search(candidate):
                    orgs.append(candidate)
                else:
                    persons.append(candidate)

            elif len(run) == 1:
                # Single cap word — use context
                prev = tokens[i - 1] if i > 0 else ""
                if _GEO_MARKERS.search(tok) or _GEO_MARKERS.search(prev):
                    locations.append(tok)
                elif _ORG_MARKERS.search(tok) or _ORG_MARKERS.search(prev):
                    orgs.append(tok)
                elif _PERSON_TITLES.search(prev):
                    persons.append(tok)
                # else: isolated capitalized word at start of sentence → skip

            i = j if j > i + 1 else i + 1

    return {
        "persons":   persons,
        "orgs":      orgs,
        "locations": locations,
    }


def entity_frequency(df: pd.DataFrame, top_n: int = 20,
                     window_days: int = 14) -> Dict[str, pd.Series]:
    """Top entities in recent N days (adaptive window to actual data range)."""
    max_date   = df["published_at"].max()
    min_date   = df["published_at"].min()
    actual_span = max(1, (max_date - min_date).days)
    effective_window = min(window_days, actual_span)

    cutoff = max_date - pd.Timedelta(days=effective_window)
    recent = df[df["published_at"] >= cutoff]
    if recent.empty:
        recent = df  # fallback: use all data

    texts = _texts_for_outlet(recent)
    ents  = extract_entities(texts)

    result = {}
    for key, lst in ents.items():
        result[key] = pd.Series(lst).value_counts().head(top_n) if lst else pd.Series(dtype=int)
    return result


# ================================================================
#  STYLE PROFILE
# ================================================================

def style_profile(df: pd.DataFrame) -> Dict:
    """
    Compute basic style metrics for an outlet corpus:
    avg title length (words), avg lead length (words), top bigrams.
    """
    titles = df["title"].dropna().astype(str)
    leads  = df["lead"].dropna().astype(str)

    title_lens = titles.str.split().str.len()
    lead_lens  = leads.str.split().str.len()

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        ngram_range=(2, 2),
        max_features=30,
        token_pattern=r"(?u)\b[а-яА-ЯёЁa-zA-Z]{3,}\b",  # no numbers/punct
        stop_words=list(stopwords),
    )
    bigrams: List[str] = []
    if len(titles) >= 5:
        try:
            vec.fit(titles)
            bigrams = list(vec.get_feature_names_out())[:10]
        except Exception:
            pass

    return {
        "avg_title_len": round(title_lens.mean(), 1) if len(title_lens) else 0,
        "avg_lead_len":  round(lead_lens.mean(), 1)  if len(lead_lens)  else 0,
        "top_bigrams":   bigrams,
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
    freq  = topic_frequency(df, labels, names)
    noise = noise_check(df, slug)
    ents  = entity_frequency(df)
    style = style_profile(df)

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
