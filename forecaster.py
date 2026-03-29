"""
Forecaster:
  1. Inertia baseline   — yesterday's topic distribution
  2. Frequency baseline — top topics last 14 days
  3. Calendar baseline  — titles from known events
  4. LLM generation     — OpenRouter model in outlet's style
"""
from collections import Counter
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    OUTLETS, FORECASTS_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, OPENROUTER_MODEL,
    FREQ_WINDOW, TARGET_DATE,
)
from etl import load_clean
from analyzer import (
    extract_topics, topic_frequency, entity_frequency,
    _get_stopwords, style_profile, _texts_for_outlet,
)
from event_calendar import get_events_for_outlet, summarize_events


# ================================================================
#  OPENROUTER (LLM) INTEGRATION
# ================================================================

def check_llm_availability() -> bool:
    """Return True if OpenRouter API key is set."""
    if not OPENROUTER_API_KEY:
        print("[llm] OPENROUTER_API_KEY not found in environment.")
        return False
    return True


def _llm_chat(prompt: str) -> str:
    """Send a single prompt to OpenRouter and return the response text."""
    from openai import OpenAI
    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=60.0,
        max_retries=1,
    )

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
            extra_headers={
                "HTTP-Referer": "https://github.com/aazhivotrev/titles_forecating", # Optional
                "X-Title": "Titles Forecasting Project", # Optional
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[llm] OpenRouter request failed: {exc}")
        return ""


# ================================================================
#  BASELINE 1 — INERTIA
# ================================================================

def inertia_forecast(df: pd.DataFrame, outlet: str,
                     target_date: datetime.date) -> List[Dict]:
    """
    Yesterday's topic set as tomorrow's forecast.
    Returns list of topic dicts.
    """
    yesterday = target_date - datetime.timedelta(days=1)
    df_yest = df[df["published_at"].dt.date == yesterday]
    if df_yest.empty:
        # fall back to last 2 days
        two_days_ago = target_date - datetime.timedelta(days=2)
        df_yest = df[df["published_at"].dt.date >= two_days_ago]

    if df_yest.empty:
        return []

    labels, names, _ = extract_topics(df_yest, outlet, n_topics=min(10, len(df_yest)))
    freq = topic_frequency(df_yest, labels, names, window=2)

    results = []
    for _, row in freq.head(10).iterrows():
        results.append({
            "method":       "inertia",
            "outlet":       outlet,
            "date":         str(target_date),
            "rubric":       row["cluster_name"].split(" / ")[0],
            "topic_label":  row["cluster_name"],
            "count_base":   int(row["count"]),
            "title":        None,
            "lead":         None,
        })
    return results


# ================================================================
#  BASELINE 2 — FREQUENCY
# ================================================================

def frequency_forecast(df: pd.DataFrame, outlet: str,
                       target_date: datetime.date,
                       window: int = FREQ_WINDOW) -> List[Dict]:
    """
    Top topics over the last `window` days.
    """
    cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=window)
    df_w   = df[df["published_at"] >= cutoff]

    if df_w.empty:
        return []

    labels, names, _ = extract_topics(df_w, outlet)
    freq = topic_frequency(df_w, labels, names, window=window)

    results = []
    for _, row in freq.head(10).iterrows():
        results.append({
            "method":       "frequency",
            "outlet":       outlet,
            "date":         str(target_date),
            "rubric":       row["cluster_name"].split(" / ")[0],
            "topic_label":  row["cluster_name"],
            "count_base":   int(row["count"]),
            "title":        None,
            "lead":         None,
        })
    return results


# ================================================================
#  BASELINE 3 — CALENDAR
# ================================================================

def _calendar_title_for_event(event: Dict) -> str:
    """Make calendar headlines closer to natural news titles than raw templates."""
    desc = str(event.get("description", "")).strip()
    if not desc:
        return ""
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc[0].upper() + desc[1:]


def calendar_forecast(outlet: str, target_date: datetime.date) -> List[Dict]:
    """
    Template headlines from known calendar events.
    """
    events = get_events_for_outlet(outlet, target_date, window_days=3)
    results = []
    for ev in events:
        rubric = ev["event_type"]
        desc   = ev["description"]
        title  = _calendar_title_for_event(ev)
        if not title:
            continue
        results.append({
            "method":       "calendar",
            "outlet":       outlet,
            "date":         str(target_date),
            "rubric":       rubric,
            "topic_label":  desc,
            "count_base":   0,
            "title":        title,
            "lead":         None,
            "event_source": ev.get("source", ""),
        })
    return results

def _get_date_meta(target_date: datetime.date, events: List[Dict]) -> str:
    """Return Russian day name and optional holiday info for context."""
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    day_name = days[target_date.weekday()]
    
    # Check if target_date itself is a holiday
    holidays = [e["description"] for e in events if e.get("event_type") == "holiday" and e.get("date") == target_date]
    if holidays:
        # Avoid duplicate "День смеха (1 апреля)" -> just info
        h_str = ", ".join(holidays)
        return f"{day_name}, {h_str}"
    
    return day_name


_GROUNDING_TOKEN_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]{3,}")
_GROUNDING_PHRASE_RE = re.compile(
    r"(?:[А-ЯЁA-Z]{2,}|[А-ЯЁA-Z][а-яёa-z]{2,})"
    r"(?:\s+(?:[А-ЯЁA-Z]{2,}|[А-ЯЁA-Z][а-яёa-z]{2,})){0,2}"
)
_GROUNDING_LEADING_WORDS = {
    "в", "во", "на", "по", "при", "для", "после", "перед", "над", "под",
    "из", "от", "до", "об", "о", "к", "ко", "что", "как", "почему",
    "когда", "где", "если",
}
_GROUNDING_NOISE_WORDS = {
    "россия", "россии", "российский", "российских", "российская", "российские",
    "сша", "украина", "украине", "украины", "область", "области", "страна",
    "страны", "мире", "мире", "москва", "москве", "рублей", "рубля", "против",
    "заявил", "заявила", "заявили", "сообщил", "сообщила", "сообщили",
    "стало", "известно", "может", "могут", "будет", "будут", "последние",
    "время", "глава", "лидер", "человек", "тысяч", "миллионов", "миллиардов",
    "ситуация", "ситуации", "вопрос", "вопросы", "данные", "рынок", "техника",
}
_GROUNDING_NOISE_PHRASES = {
    "в россии", "в сша", "на украине", "над россией", "в москве",
    "кубка россии", "власти россии", "пользователи telegram",
}


# ================================================================
#  LLM — STYLE EXAMPLES SELECTOR
# ================================================================

def _pick_style_examples(df: pd.DataFrame, n: int = 5) -> List[str]:
    """Pick n recent example titles (and leads if available) for the LLM prompt."""
    recent = df.sort_values("published_at", ascending=False).head(20)
    examples = []
    for _, row in recent.iterrows():
        title = str(row.get("title") or "").strip()
        lead  = str(row.get("lead") or "").strip()
        if title and title.lower() not in ("nan", "none", ""):
            sample = f"Заголовок: {title}"
            if lead and lead.lower() not in ("nan", "none", ""):
                sample += f"\nЛид: {lead[:200]}"
            examples.append(sample)
        if len(examples) >= n:
            break
    return examples


def _pick_recent_headlines(df: pd.DataFrame, n: int = 6) -> List[str]:
    """Pick recent headlines only, used as topic grounding in the prompt."""
    recent = df.sort_values("published_at", ascending=False).head(30)
    headlines = []
    for _, row in recent.iterrows():
        title = str(row.get("title") or "").strip()
        if title and title.lower() not in ("nan", "none", ""):
            headlines.append(title)
        if len(headlines) >= n:
            break
    return headlines


def _pick_headlines_from_rows(df_rows: pd.DataFrame, n: int = 6) -> List[str]:
    """Collect headlines from rows preserving their current ranking/order."""
    headlines = []
    for _, row in df_rows.iterrows():
        title = str(row.get("title") or "").strip()
        if title and title.lower() not in ("nan", "none", ""):
            headlines.append(title)
        if len(headlines) >= n:
            break
    return headlines


def _pick_style_examples_from_rows(df_rows: pd.DataFrame, n: int = 5) -> List[str]:
    """Build prompt examples from a scored/retrieved subset of rows."""
    examples = []
    for _, row in df_rows.iterrows():
        title = str(row.get("title") or "").strip()
        lead = str(row.get("lead") or "").strip()
        if title and title.lower() not in ("nan", "none", ""):
            sample = f"Заголовок: {title}"
            if lead and lead.lower() not in ("nan", "none", ""):
                sample += f"\nЛид: {lead[:200]}"
            examples.append(sample)
        if len(examples) >= n:
            break
    return examples


def _normalize_grounding_phrase(text: str) -> str:
    """Trim punctuation/noisy leading words and drop overly generic phrases."""
    cleaned = re.sub(r"[\"'`«»(),.:;!?]", " ", str(text))
    tokens = cleaned.split()
    while tokens and tokens[0].lower() in _GROUNDING_LEADING_WORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in _GROUNDING_LEADING_WORDS:
        tokens = tokens[:-1]
    if not tokens:
        return ""

    phrase = " ".join(tokens).strip()
    lower = phrase.lower()
    if lower in _GROUNDING_NOISE_PHRASES:
        return ""
    if len(tokens) == 1 and lower in _GROUNDING_NOISE_WORDS:
        return ""
    return phrase


def _headline_phrase_candidates(headlines: List[str], top_k: int = 8) -> List[str]:
    """Extract repeated capitalized phrases from recent headlines."""
    counts: Counter = Counter()
    for title in headlines:
        for match in _GROUNDING_PHRASE_RE.findall(str(title)):
            phrase = _normalize_grounding_phrase(match)
            if phrase:
                counts[phrase] += 1

    ranked = sorted(counts.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    phrases = []
    for phrase, count in ranked:
        words = phrase.split()
        if len(words) == 1 and not words[0].isupper() and count < 2:
            continue
        phrases.append(phrase)
        if len(phrases) >= top_k:
            break
    return phrases


def _headline_term_candidates(headlines: List[str], top_k: int = 10) -> List[str]:
    """Extract recurring content words and bigrams from recent headlines."""
    stopwords = _get_stopwords() | _GROUNDING_NOISE_WORDS
    counts: Counter = Counter()

    for title in headlines:
        tokens = [
            token.lower()
            for token in _GROUNDING_TOKEN_RE.findall(str(title))
            if token.lower() not in stopwords
        ]
        counts.update(tokens)
        counts.update(
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
            if left != right
        )

    ranked = []
    for term, count in counts.items():
        words = term.split()
        if count < 2 and len(words) == 1:
            continue
        if all(word in _GROUNDING_NOISE_WORDS for word in words):
            continue
        ranked.append((count, len(words), term))

    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [term for _, _, term in ranked[:top_k]]


def _build_topic_vectorizer(df: pd.DataFrame) -> Tuple[Optional[TfidfVectorizer], Optional[np.ndarray]]:
    """Fit one outlet-level TF-IDF model used to retrieve tighter topic context."""
    texts = _texts_for_outlet(df)
    if not texts:
        return None, None

    stopwords = _get_stopwords() | _GROUNDING_NOISE_WORDS
    min_df = 2 if len(texts) >= 100 else 1
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[а-яА-ЯёЁa-zA-Z]{3,}\b",
        min_df=min_df,
        max_df=0.7,
        max_features=5000,
        stop_words=list(stopwords),
    )
    try:
        matrix = vec.fit_transform(texts)
        return vec, matrix
    except ValueError:
        return None, None


def _topic_signature_terms_from_rows(
    df_topic: pd.DataFrame,
    vec: Optional[TfidfVectorizer],
    top_k: int = 10,
) -> List[str]:
    """Get the most distinctive terms inside a topic cluster using outlet TF-IDF."""
    if vec is None or df_topic.empty:
        return []

    topic_texts = _texts_for_outlet(df_topic)
    if not topic_texts:
        return []

    try:
        topic_matrix = vec.transform(topic_texts)
    except Exception:
        return []

    weights = np.asarray(topic_matrix.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    ranked_idx = weights.argsort()[::-1]

    signature_terms: List[str] = []
    seen = set()
    for idx in ranked_idx:
        if weights[idx] <= 0:
            break
        term = str(terms[idx]).strip()
        lower = term.lower()
        if not term or lower in seen:
            continue
        parts = term.split()
        if all(part in _GROUNDING_NOISE_WORDS for part in parts):
            continue
        seen.add(lower)
        signature_terms.append(term)
        if len(signature_terms) >= top_k:
            break
    return signature_terms


def _retrieve_topic_rows(
    df_topic: pd.DataFrame,
    signature_terms: List[str],
    vec: Optional[TfidfVectorizer],
    top_k: int = 8,
) -> pd.DataFrame:
    """Pick the most on-topic historical rows inside a broad cluster."""
    if df_topic.empty:
        return df_topic.head(0)

    ranked = df_topic.sort_values("published_at", ascending=False).copy()
    if vec is None or not signature_terms:
        return ranked.head(top_k)

    try:
        topic_texts = _texts_for_outlet(ranked)
        topic_matrix = vec.transform(topic_texts)
        query_vec = vec.transform([" ".join(signature_terms[:8])])
        sims = cosine_similarity(topic_matrix, query_vec).ravel()
    except Exception:
        return ranked.head(top_k)

    if "published_at" in ranked and len(ranked) > 1:
        order = np.arange(len(ranked), 0, -1, dtype=float)
        recency = order / order.max()
    else:
        recency = np.ones(len(ranked), dtype=float)

    ranked = ranked.assign(_retrieval_score=sims + recency * 0.15)
    ranked = ranked.sort_values(["_retrieval_score", "published_at"], ascending=[False, False])
    return ranked.head(top_k).drop(columns=["_retrieval_score"], errors="ignore")


def _recent_topic_window(df_topic: pd.DataFrame, max_days: int = 30, min_rows: int = 20) -> pd.DataFrame:
    """Prefer fresh topic history so retrieval doesn't drift to stale old stories."""
    if df_topic.empty or "published_at" not in df_topic:
        return df_topic

    latest = df_topic["published_at"].max()
    cutoff = latest - pd.Timedelta(days=max_days)
    recent = df_topic[df_topic["published_at"] >= cutoff]
    return recent if len(recent) >= min_rows else df_topic


def _clean_grounding_entities(raw_entities: List[str], recent_headlines: List[str],
                              top_k: int = 8) -> List[str]:
    """Prefer headline-derived phrases, then add cleaned entity candidates."""
    candidates: List[str] = []
    seen = set()
    headline_token_counts: Counter = Counter(
        token.lower()
        for title in recent_headlines
        for token in _GROUNDING_TOKEN_RE.findall(str(title))
    )

    def add(item: str) -> None:
        normalized = _normalize_grounding_phrase(item)
        if not normalized:
            return
        words = normalized.split()
        if len(words) == 1 and not words[0].isupper():
            if headline_token_counts[words[0].lower()] < 2:
                return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    for phrase in _headline_phrase_candidates(recent_headlines, top_k=top_k):
        add(phrase)
        if len(candidates) >= top_k:
            return candidates

    for entity in raw_entities:
        add(str(entity).strip())
        if len(candidates) >= top_k:
            break
    return candidates


def _topic_keywords(topic_label: str, top_entities: List[str],
                    recent_headlines: List[str], top_k: int = 10) -> List[str]:
    """Build a more topic-specific keyword list for grounding and filtering."""
    stopwords = _get_stopwords()
    seen = set()
    keywords: List[str] = []

    def add(term: str) -> None:
        text = str(term).strip()
        lower = text.lower()
        if not text or lower in seen:
            return
        if len(text.split()) == 1 and (lower in stopwords or lower in _GROUNDING_NOISE_WORDS):
            return
        seen.add(lower)
        keywords.append(text)

    for entity in top_entities:
        add(entity)
        for token in _GROUNDING_TOKEN_RE.findall(str(entity).lower()):
            add(token)
        if len(keywords) >= top_k:
            return keywords[:top_k]

    for term in _headline_term_candidates(recent_headlines, top_k=top_k * 2):
        add(term)
        if len(keywords) >= top_k:
            return keywords[:top_k]

    for token in _GROUNDING_TOKEN_RE.findall(topic_label.lower()):
        if token in stopwords or token in _GROUNDING_NOISE_WORDS:
            continue
        add(token)
        if len(keywords) >= top_k:
            break
    return keywords[:top_k]


# ================================================================
#  LLM — PROMPT BUILDER
# ================================================================

def _build_llm_prompt(
    outlet: str,
    target_date: datetime.date,
    date_meta: str,
    topic_label: str,
    topic_metrics: str,
    events_summary: str,
    style_examples: List[str],
    recent_headlines: List[str],
    top_entities: List[str],
    topic_keywords: List[str],
    has_lead: bool,
    requested_items: int = 4,
) -> str:
    outlet_name  = OUTLETS[outlet]["name"]
    date_str     = target_date.strftime("%d.%m.%Y")
    examples_str = "\n\n".join(style_examples) if style_examples else "(примеры недоступны)"
    entities_str = ", ".join(top_entities[:10]) if top_entities else "нет данных"
    keywords_str = ", ".join(topic_keywords[:10]) if topic_keywords else "нет данных"
    headlines_str = "\n".join(f"- {h}" for h in recent_headlines[:6]) if recent_headlines else "(связанные заголовки недоступны)"

    lead_req = "ЗАГОЛОВОК и ЛИД (1-2 предложения, до 50 слов)" if has_lead else "только ЗАГОЛОВОК"
    format_req = (
        "1. ЗАГОЛОВОК: ...\n   ЛИД: ...\n2. ЗАГОЛОВОК: ...\n   ЛИД: ..."
        if has_lead else
        "1. ЗАГОЛОВОК: ...\n2. ЗАГОЛОВОК: ..."
    )

    return f"""# CONTEXT
Издание: «{outlet_name}».
Целевая дата публикации: {date_str} ({date_meta}).
Смысловая тематика (определена алгоритмом): «{topic_label}» (внимание: улови суть темы из этих слов, избегай их точного бездумного копирования).
{topic_metrics}
Ключевые слова темы: {keywords_str}
Известные календарные события:
{events_summary}
Главные действующие лица и организации темы: {entities_str}.
Связанные недавние заголовки по теме:
{headlines_str}

# OBJECTIVE
Сгенерируй {requested_items} правдоподобных новостных сюжета на указанную дату, которые могли бы естественным образом появиться в издании «{outlet_name}».
Используй актуальную повестку (календарные события и реальных действующих лиц) вместо абстрактных или выдуманных (фантастических) событий.
Каждый сюжет должен быть явно связан хотя бы с одной сущностью или ключевым словом из списка выше.
Предпочитай развитие уже наблюдаемой повестки, а не случайные новые ветки.

# STYLE
Новостная заметка. Используй структуру предложений, лексику и подачу, характерные для этого издания.
Ориентируйся на этот срез реальных недавних публикаций:
{examples_str}

# TONE
Профессиональный тон, полностью копирующий оригинальный tone-of-voice рассматриваемого медиа (официоз, деловая аналитика и т.д. — проанализируй это из примеров).

# AUDIENCE
Постоянные читатели источника «{outlet_name}», привыкшие к стандартам этой редакции.

# RESPONSE
Выведи ровно {requested_items} пронумерованных пункта.
Для каждой новости напиши {lead_req}.
Выводи только результат, без вводных или заключительных приветствий.
Не повторяй один и тот же сюжет разными словами.
Используй строго следующий формат:
{format_req}
"""


# ================================================================
#  LLM — PARSE RESPONSE
# ================================================================

def _parse_llm_response(text: str, outlet: str, target_date: datetime.date,
                        topic_label: str) -> List[Dict]:
    """Parse numbered list from LLM output into structured records."""
    results = []
    # Split by numbered entries
    blocks = re.split(r"\n\s*\d+\.\s+", "\n" + text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        title = ""
        lead  = ""
        m_t = re.search(r"ЗАГОЛОВОК\s*:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
        m_l = re.search(r"ЛИД\s*:\s*(.+?)(?:\n\d+\.|$)", block, re.IGNORECASE | re.DOTALL)
        if m_t:
            title = m_t.group(1).strip()
        if m_l:
            lead  = m_l.group(1).strip().replace("\n", " ")
        if not title and block:
            # Fallback: first non-empty line
            title = block.splitlines()[0].strip()

        if title:
            results.append({
                "method":      "llm",
                "outlet":      outlet,
                "date":        str(target_date),
                "rubric":      topic_label.split(" / ")[0],
                "topic_label": topic_label,
                "count_base":  0,
                "title":       title,
                "lead":        lead or None,
            })
    return results


def _filter_llm_results(
    results: List[Dict],
    topic_keywords: List[str],
    top_entities: List[str],
    recent_headlines: Optional[List[str]] = None,
    min_keep: int = 3,
) -> List[Dict]:
    """
    Keep headlines that stay close to the target topic, then deduplicate.
    """
    if not results:
        return []

    keyword_set = {
        kw.lower() for kw in topic_keywords
        if isinstance(kw, str) and kw.strip()
    }
    entity_set = {
        ent.lower() for ent in top_entities
        if isinstance(ent, str) and ent.strip()
    }
    recent_term_set = {
        term.lower() for term in _headline_term_candidates(recent_headlines or [], top_k=12)
        if isinstance(term, str) and term.strip()
    }

    scored = []
    seen_titles = []
    for item in results:
        title = str(item.get("title") or "").strip()
        lead = str(item.get("lead") or "").strip()
        text = f"{title} {lead}".lower()

        keyword_hits = sum(1 for kw in keyword_set if kw in text)
        entity_hits = sum(1 for ent in entity_set if ent in text)
        recent_hits = sum(1 for term in recent_term_set if term in text)
        score = entity_hits * 3 + keyword_hits * 2 + recent_hits
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [item for score, item in scored if score > 0]
    if len(filtered) < min_keep:
        filtered = [item for _, item in scored[:max(min_keep, len(filtered))]]

    unique_results: List[Dict] = []
    for item in filtered:
        title = str(item.get("title") or "").strip().lower()
        if not title:
            continue
        if any(title == prev for prev in seen_titles):
            continue
        seen_titles.append(title)
        unique_results.append(item)
    return unique_results


# ================================================================
#  LLM — MAIN GEN FUNCTION
# ================================================================

def llm_forecast(
    df: pd.DataFrame,
    labels: pd.Series,
    outlet: str,
    target_date: datetime.date,
    topic_info: List[Dict],
    events_summary: str,
    events_list: List[Dict] = None,
    n_topics: int = 3,
) -> List[Dict]:
    """
    Generate LLM-based forecasts for top N topics.
    """
    if not check_llm_availability():
        print(f"  [llm] Skipping LLM for {outlet} — API key missing")
        return []

    has_lead = OUTLETS[outlet]["has_lead"]
    
    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    date_meta = _get_date_meta(target_date, events_list or [])
    topic_vec, _ = _build_topic_vectorizer(df)
    
    all_results = []
    for info in topic_info[:n_topics]:
        label = info["name"]
        cluster_id = info["cluster"]
        
        # Context strictly isolated to current topic
        df_topic = df_clustered[df_clustered["cluster"] == cluster_id]
        df_topic_recent = _recent_topic_window(df_topic, max_days=30, min_rows=20)

        signature_terms = _topic_signature_terms_from_rows(df_topic_recent, topic_vec, top_k=10)
        support_rows = _retrieve_topic_rows(df_topic_recent, signature_terms, topic_vec, top_k=8)
        support_headlines = _pick_headlines_from_rows(support_rows, n=8)

        phrase_entities = _headline_phrase_candidates(support_headlines, top_k=8)
        if phrase_entities:
            top_entities = phrase_entities
        else:
            topic_entities_data = entity_frequency(support_rows if not support_rows.empty else df_topic)
            top_persons = topic_entities_data.get("persons", pd.Series()).index.tolist()[:7]
            top_orgs = topic_entities_data.get("orgs", pd.Series()).index.tolist()[:7]
            raw_entities = top_persons + top_orgs
            top_entities = _clean_grounding_entities(raw_entities, support_headlines, top_k=8)

        topic_keywords = _topic_keywords(label, top_entities + signature_terms, support_headlines)

        style_examples = _pick_style_examples_from_rows(support_rows, n=4)
        if len(style_examples) < 6:
            style_examples.extend(_pick_style_examples(df, n=6 - len(style_examples)))

        metrics = (f"Популярность темы в последнее время: {info['count']} новостей "
                   f"({info['pct']}% от общего потока).")
        
        print(f"  [llm] {outlet} | topic: {label[:60]}...")
        prompt = _build_llm_prompt(
            outlet, target_date, date_meta, label, metrics,
            events_summary, style_examples, support_headlines, top_entities,
            topic_keywords, has_lead=has_lead, requested_items=4,
        )
        try:
            raw = _llm_chat(prompt)
            parsed = _parse_llm_response(raw, outlet, target_date, label)
            filtered = _filter_llm_results(
                parsed,
                topic_keywords,
                top_entities,
                recent_headlines=support_headlines,
                min_keep=3,
            )
            all_results.extend(filtered)
        except Exception as exc:
            print(f"  [llm] Error for {outlet}/{label}: {exc}")

    return all_results


# ================================================================
#  COMBINE FORECAST
# ================================================================

def combine_forecast(
    outlet: str,
    target_date: datetime.date = TARGET_DATE,
    use_llm: bool = True,
) -> Dict:
    """
    Run all forecasting methods for one outlet and return combined report.
    """
    df = load_clean(outlet)
    if df.empty:
        print(f"  [forecast] No clean data for {outlet}")
        return {}

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    outlet_name = OUTLETS[outlet]["name"]
    print(f"\n[forecast] {outlet_name} → {target_date}")

    # Topic analysis on full history
    labels, names, _ = extract_topics(df, outlet)
    freq  = topic_frequency(df, labels, names, window=FREQ_WINDOW)
    top_topics = freq["cluster_name"].head(5).tolist()

    # Events context
    events = get_events_for_outlet(outlet, target_date, window_days=3)
    events_summary = summarize_events(events)

    # Baselines
    inertia  = inertia_forecast(df, outlet, target_date)
    freq_fc  = frequency_forecast(df, outlet, target_date)
    calendar = calendar_forecast(outlet, target_date)

    # Prepare topic info for LLM
    topic_info = freq.head(5).to_dict("records")
    # Add cluster_name to name if missing
    for ti in topic_info:
        ti["name"] = ti.get("cluster_name", "unknown")

    # LLM
    llm_results: List[Dict] = []
    if use_llm:
        llm_results = llm_forecast(df, labels, outlet, target_date,
                                   topic_info, events_summary, events)

    all_preds = inertia + freq_fc + calendar + llm_results

    report = {
        "outlet":           outlet,
        "outlet_name":      outlet_name,
        "target_date":      str(target_date),
        "top_topics":       top_topics,
        "events_context":   events_summary,
        "predictions":      all_preds,
        "llm_count":        len(llm_results),
        "calendar_count":   len(calendar),
        "baseline_inertia": len(inertia),
        "baseline_freq":    len(freq_fc),
    }
    return report


def forecast_all(
    slugs: List[str] = None,
    target_date: datetime.date = TARGET_DATE,
    use_llm: bool = True,
) -> Dict[str, Dict]:
    """
    Forecast for all outlets. Saves JSON to data/forecasts/.
    """
    from config import OUTLET_SLUGS
    if slugs is None:
        slugs = OUTLET_SLUGS

    all_reports: Dict[str, Dict] = {}
    for slug in slugs:
        report = combine_forecast(slug, target_date, use_llm=use_llm)
        all_reports[slug] = report

    # Write combined JSON
    date_str = str(target_date)
    out_path = os.path.join(FORECASTS_DIR, f"forecast_{date_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[forecast] Saved → {out_path}")
    return all_reports
