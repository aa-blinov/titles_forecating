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
from sklearn.cluster import KMeans
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


def _forecast_run_stamp(now: Optional[datetime.datetime] = None) -> str:
    """Return a timestamp suffix for archived forecast artifacts."""
    return (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


def _forecast_output_paths(
    target_date: datetime.date,
    run_stamp: str,
    forecast_strategy: str,
    forecast_profile: str,
) -> Tuple[str, str]:
    """Return timestamped JSON/XLSX paths for a forecast run."""
    strategy_part = re.sub(r"[^a-z0-9_]+", "_", str(forecast_strategy).lower()).strip("_") or "llm"
    profile_part = re.sub(r"[^a-z0-9_]+", "_", str(forecast_profile).lower()).strip("_") or "default"
    base_name = f"forecast_{target_date}_{strategy_part}_{profile_part}_{run_stamp}"
    return (
        os.path.join(FORECASTS_DIR, f"{base_name}.json"),
        os.path.join(FORECASTS_DIR, f"{base_name}.xlsx"),
    )


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


def _inertia_window_rows(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    """Return yesterday's rows, or a short fallback window if yesterday is empty."""
    yesterday = target_date - datetime.timedelta(days=1)
    df_yest = df[df["published_at"].dt.date == yesterday]
    if not df_yest.empty:
        return df_yest

    fallback_start = target_date - datetime.timedelta(days=2)
    return df[df["published_at"].dt.date >= fallback_start]


def _event_signal_terms(events: List[Dict], top_k: int = 18) -> List[str]:
    """Extract calendar terms used to boost related topics in hybrid mode."""
    if not events:
        return []

    texts = [
        str(event.get("description") or "").strip()
        for event in events
        if str(event.get("description") or "").strip()
    ]
    if not texts:
        return []

    phrases = _headline_phrase_candidates(texts, top_k=max(4, top_k // 2))
    terms = _headline_term_candidates(texts, top_k=top_k)

    unique_terms: List[str] = []
    seen = set()
    for term in phrases + terms:
        lower = str(term).lower().strip()
        if not lower or lower in seen:
            continue
        seen.add(lower)
        unique_terms.append(str(term).strip())
        if len(unique_terms) >= top_k:
            break
    return unique_terms


_RU_MONTHS = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

_RU_WEEKDAYS = {
    0: "понедельник",
    1: "вторник",
    2: "среда",
    3: "четверг",
    4: "пятница",
    5: "суббота",
    6: "воскресенье",
}

_FORWARD_MARKERS = [
    "планируется", "запланирован", "запланирована", "запланировано",
    "состоится", "пройдет", "пройдёт", "ожидается", "обсудит", "обсудят",
    "рассмотрит", "рассмотрят", "назначено на", "намечено на", "намечен на",
    "вступит в силу", "будет опублик", "будут опублик", "будет представлен",
    "будут представлены", "должен состояться", "должна состояться",
    "подготовит", "объявит", "проведет", "проведёт", "переговоры пройдут",
    "заседание пройдет", "заседание пройдёт", "может состояться",
    "получат", "пройдут", "начнется", "начнётся", "вступают в силу",
]


def _target_date_terms(target_date: datetime.date) -> Tuple[List[str], List[str]]:
    """Return direct date strings and weekday phrases for forward-looking retrieval."""
    month_name = _RU_MONTHS[target_date.month]
    weekday_name = _RU_WEEKDAYS[target_date.weekday()]
    day = target_date.day

    direct_terms = [
        f"{day} {month_name}",
        f"{day:02d} {month_name}",
        f"{day}.{target_date.month}",
        f"{day:02d}.{target_date.month:02d}",
        f"{day:02d}.{target_date.month:02d}.{target_date.year}",
        f"{day}.{target_date.month}.{target_date.year}",
    ]
    weekday_terms = [
        f"в {weekday_name}",
        f"во {weekday_name}" if weekday_name.startswith("в") else "",
        f"в этот {weekday_name}",
        f"в ближайший {weekday_name}",
        f"в ближайшую {weekday_name}" if weekday_name.endswith("а") else "",
    ]
    weekday_terms = [term for term in weekday_terms if term]
    return direct_terms, weekday_terms


def _week_index(day: datetime.date) -> Tuple[int, int]:
    iso = day.isocalendar()
    return int(iso.year), int(iso.week)


def _forward_temporal_hits(
    text: str,
    published_date: datetime.date,
    target_date: datetime.date,
    direct_terms: List[str],
    weekday_terms: List[str],
) -> Dict[str, List[str]]:
    """Resolve soft temporal phrases against the target date and publication date."""
    lower = str(text).lower()
    days_until = (target_date - published_date).days
    month_name = _RU_MONTHS[target_date.month]

    direct_hits = [term for term in direct_terms if term in lower]

    weekday_hits: List[str] = []
    if 0 < days_until <= 7:
        weekday_hits = [term for term in weekday_terms if term in lower]

    relative_terms = []
    if days_until == 1:
        relative_terms.append("завтра")
    elif days_until == 2:
        relative_terms.append("послезавтра")
    relative_hits = [term for term in relative_terms if term in lower]

    current_week = _week_index(published_date)
    target_week = _week_index(target_date)
    week_terms: List[str] = []
    if current_week == target_week and 0 < days_until <= 6:
        week_terms.extend(["на этой неделе", "до конца недели"])
    else:
        pub_next_week = _week_index(published_date + datetime.timedelta(days=7))
        if target_week == pub_next_week and 0 < days_until <= 13:
            week_terms.extend(["на следующей неделе", "в начале следующей недели"])
    week_hits = [term for term in week_terms if term in lower]

    month_terms = [
        f"в {month_name}",
        f"в начале {month_name}",
        f"в первых числах {month_name}",
        f"в первой половине {month_name}",
    ]
    if target_date.day <= 10:
        month_terms.append("в начале месяца")
        if published_date.month != target_date.month:
            month_terms.append("в начале следующего месяца")
            month_terms.append("в следующем месяце")
    month_hits = [term for term in month_terms if term in lower]

    return {
        "direct_hits": sorted(set(direct_hits)),
        "weekday_hits": sorted(set(weekday_hits)),
        "relative_hits": sorted(set(relative_hits)),
        "week_hits": sorted(set(week_hits)),
        "month_hits": sorted(set(month_hits)),
    }


def _extract_forward_signal_rows(
    df_clustered: pd.DataFrame,
    target_date: datetime.date,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """Find recent rows that explicitly look ahead to the target date."""
    if df_clustered.empty or "published_at" not in df_clustered:
        return df_clustered.head(0)

    lo = pd.Timestamp(target_date) - pd.Timedelta(days=lookback_days)
    hi = pd.Timestamp(target_date)
    recent = df_clustered[
        (df_clustered["published_at"] >= lo) & (df_clustered["published_at"] < hi)
    ].copy()
    if recent.empty:
        return recent

    direct_terms, weekday_terms = _target_date_terms(target_date)
    rows: List[Dict] = []
    for _, row in recent.iterrows():
        title = str(row.get("title") or "").strip()
        lead = str(row.get("lead") or "").strip()
        text = " ".join(part for part in [title, lead] if part).lower()
        if not text:
            continue

        published_at = row.get("published_at")
        if pd.isna(published_at):
            continue
        published_date = pd.Timestamp(published_at).date()
        temporal_hits = _forward_temporal_hits(
            text,
            published_date,
            target_date,
            direct_terms,
            weekday_terms,
        )
        direct_hits = temporal_hits["direct_hits"]
        weekday_hits = temporal_hits["weekday_hits"]
        relative_hits = temporal_hits["relative_hits"]
        week_hits = temporal_hits["week_hits"]
        month_hits = temporal_hits["month_hits"]
        forward_hits = [term for term in _FORWARD_MARKERS if term in text]

        temporal_hit_count = sum(
            len(hits)
            for hits in [direct_hits, weekday_hits, relative_hits, week_hits, month_hits]
        )
        qualifies = (
            bool(direct_hits)
            or bool(relative_hits)
            or bool(weekday_hits)
            or bool(week_hits)
            or bool(month_hits)
            or len(forward_hits) >= 2
            or (temporal_hit_count >= 1 and bool(forward_hits))
        )
        if not qualifies:
            continue

        signal_score = (
            len(set(direct_hits)) * 3.0
            + len(set(weekday_hits)) * 1.5
            + len(set(relative_hits)) * 2.0
            + len(set(week_hits)) * 1.75
            + len(set(month_hits)) * 1.5
            + len(set(forward_hits)) * 1.25
        )
        if direct_hits and forward_hits:
            signal_score += 1.5
        if (relative_hits or week_hits or month_hits) and forward_hits:
            signal_score += 1.0

        rows.append({
            "cluster": row.get("cluster"),
            "published_at": published_at,
            "title": title,
            "lead": lead,
            "signal_score": float(signal_score),
            "direct_hits": sorted(set(direct_hits)),
            "weekday_hits": sorted(set(weekday_hits)),
            "relative_hits": sorted(set(relative_hits)),
            "week_hits": sorted(set(week_hits)),
            "month_hits": sorted(set(month_hits)),
            "forward_hits": sorted(set(forward_hits)),
            "has_direct_target": bool(direct_hits),
        })

    if not rows:
        return recent.head(0)

    df_rows = pd.DataFrame(rows)
    return df_rows.sort_values(["signal_score", "published_at"], ascending=[False, False])


def _forward_signal_stats(
    df_clustered: pd.DataFrame,
    target_date: datetime.date,
    top_examples: int = 3,
) -> Dict:
    """Aggregate forward-looking mention stats per cluster for the forward_look profile."""
    df_forward = _extract_forward_signal_rows(df_clustered, target_date, lookback_days=14)
    if df_forward.empty:
        return {}

    stats: Dict = {}
    for cluster_id, group in df_forward.groupby("cluster"):
        group = group.sort_values(["signal_score", "published_at"], ascending=[False, False])
        direct_target_hits = int(group["has_direct_target"].sum())
        forward_count = int(len(group))
        max_signal = float(group["signal_score"].max())
        forward_score = (
            min(direct_target_hits, 3) * 2.5
            + min(forward_count, 4) * 1.25
            + min(max_signal, 6.0)
        )
        examples = group["title"].dropna().astype(str).head(top_examples).tolist()
        marker_hits = []
        seen = set()
        for hits in (
            group["forward_hits"].tolist()
            + group["direct_hits"].tolist()
            + group["relative_hits"].tolist()
            + group["week_hits"].tolist()
            + group["month_hits"].tolist()
        ):
            for hit in hits:
                if hit not in seen:
                    seen.add(hit)
                    marker_hits.append(hit)
                if len(marker_hits) >= 6:
                    break
            if len(marker_hits) >= 6:
                break

        stats[cluster_id] = {
            "forward_signal_score": float(forward_score),
            "forward_signal_count": forward_count,
            "forward_target_hits": direct_target_hits,
            "forward_examples": examples,
            "forward_markers": marker_hits,
        }
    return stats


def _apply_forward_look_profile(
    topic_info: List[Dict],
    forward_stats: Dict,
    top_n: int = 5,
) -> List[Dict]:
    """Boost topics that already contain forward-looking mentions of the target date."""
    profiled: List[Dict] = []
    for info in topic_info:
        item = dict(info)
        stat = forward_stats.get(item.get("cluster"), {})
        forward_score = float(stat.get("forward_signal_score", 0.0))
        item["forward_signal_score"] = forward_score
        item["forward_signal_count"] = int(stat.get("forward_signal_count", 0))
        item["forward_target_hits"] = int(stat.get("forward_target_hits", 0))
        item["forward_examples"] = list(stat.get("forward_examples", []))
        item["forward_markers"] = list(stat.get("forward_markers", []))
        item["ensemble_score"] = float(item.get("ensemble_score", 0.0)) + forward_score

        reasons = list(item.get("selection_reasons", []))
        if item["forward_target_hits"]:
            reasons.append(
                f"в корпусе есть прямые упоминания целевой даты ({item['forward_target_hits']})"
            )
        elif item["forward_signal_count"]:
            reasons.append(
                f"в корпусе есть анонсы и forward-looking маркеры ({item['forward_signal_count']})"
            )
        item["selection_reasons"] = reasons
        item["forecast_profile"] = "forward_look"
        profiled.append(item)

    profiled.sort(
        key=lambda item: (
            float(item.get("ensemble_score", 0.0)),
            float(item.get("forward_signal_score", 0.0)),
            int(item.get("forward_target_hits", 0)),
            int(item.get("forward_signal_count", 0)),
            float(item.get("outlet_profile_score", 0.0)),
        ),
        reverse=True,
    )
    return profiled[:top_n]


def _outlet_profile_signal(text: str, outlet: str) -> Tuple[float, List[str], List[str]]:
    """Return outlet-specific bonus/penalty from lexical profile markers."""
    profile = _OUTLET_PROFILE_PRIORS.get(outlet, {})
    if not profile:
        return 0.0, [], []

    lower = str(text).lower()

    def _hits(terms: List[str]) -> List[str]:
        matched = []
        seen = set()
        for term in terms:
            if term in lower and term not in seen:
                seen.add(term)
                matched.append(term)
        return matched

    prefer_hits = _hits(profile.get("prefer", []))
    avoid_hits = _hits(profile.get("avoid", []))
    hard_avoid_hits = _hits(profile.get("hard_avoid", []))
    prefer_weight = float(profile.get("prefer_weight", 1.35))
    avoid_weight = float(profile.get("avoid_weight", 1.85))
    score = (
        len(prefer_hits) * prefer_weight
        - len(avoid_hits) * avoid_weight
        - len(hard_avoid_hits) * 2.75
    )
    return float(score), prefer_hits, avoid_hits


def _outlet_story_gate(text: str, outlet: str) -> Tuple[float, List[str], List[str]]:
    """Return a strong outlet-specific penalty for structurally off-brand stories."""
    lower = str(text).lower()
    def _hits(terms: List[str]) -> List[str]:
        matched = []
        seen = set()
        for term in terms:
            if term in lower and term not in seen:
                seen.add(term)
                matched.append(term)
        return matched

    if outlet == "kommersant":
        military_hits = _hits(_KOMMERSANT_MILITARY_MARKERS)
        business_hits = _hits(_KOMMERSANT_BUSINESS_POLICY_MARKERS)
        if military_hits and not business_hits:
            return -8.0, military_hits, business_hits
        return 0.0, military_hits, business_hits

    if outlet == "lenta":
        conflict_hits = _hits(_LENTA_CONFLICT_MARKERS)
        bureaucratic_hits = _hits(_LENTA_BUREAUCRATIC_MARKERS)
        if bureaucratic_hits and not conflict_hits:
            return -5.5, bureaucratic_hits, conflict_hits
        return 0.0, bureaucratic_hits, conflict_hits

    if outlet == "interfax":
        sensational_hits = _hits(_INTERFAX_SENSATIONAL_MARKERS)
        official_hits = _hits(_INTERFAX_OFFICIAL_MARKERS)
        if sensational_hits and not official_hits:
            return -5.5, sensational_hits, official_hits
        return 0.0, sensational_hits, official_hits

    return 0.0, [], []


def _prepare_llm_topic_info(
    df: pd.DataFrame,
    labels: pd.Series,
    freq: pd.DataFrame,
    outlet: str,
    target_date: datetime.date,
    events: List[Dict],
    strategy: str = "llm",
    profile: str = "default",
    top_n: int = 5,
) -> List[Dict]:
    """Build topic candidates for LLM generation under standard or hybrid strategy."""
    if freq.empty:
        return []

    base_info: List[Dict] = []
    for rank, (_, row) in enumerate(freq.head(top_n).iterrows(), start=1):
        count = int(row.get("count", 0))
        pct = float(row.get("pct", 0.0))
        base_info.append({
            "cluster": row["cluster"],
            "cluster_name": row["cluster_name"],
            "name": row["cluster_name"],
            "count": count,
            "pct": pct,
            "freq_rank": rank,
            "freq_score": max(0.0, 8.0 - float(rank - 1)) + pct / 25.0,
            "inertia_count": 0,
            "event_hits": 0,
            "event_score": 0.0,
            "outlet_profile_score": 0.0,
            "outlet_prefer_hits": [],
            "outlet_avoid_hits": [],
            "forward_signal_score": 0.0,
            "forward_signal_count": 0,
            "forward_target_hits": 0,
            "forward_examples": [],
            "forward_markers": [],
            "ensemble_score": max(0.0, 8.0 - float(rank - 1)) + pct / 25.0,
            "selection_reasons": [f"частотная база {count} новостей ({pct:.1f}%)"],
            "topic_strategy": "llm",
            "forecast_profile": "default",
        })

    df_clustered = df.copy()
    df_clustered["cluster"] = labels
    forward_stats = _forward_signal_stats(df_clustered, target_date) if profile == "forward_look" else {}

    if strategy != "hybrid":
        if profile == "forward_look":
            return _apply_forward_look_profile(base_info, forward_stats, top_n=top_n)
        return base_info

    inertia_rows = _inertia_window_rows(df_clustered, target_date)
    inertia_counts: Counter = Counter(
        inertia_rows["cluster"].tolist()
    ) if not inertia_rows.empty else Counter()
    event_terms = _event_signal_terms(events, top_k=18)

    hybrid_info: List[Dict] = []
    for rank, (_, row) in enumerate(freq.iterrows(), start=1):
        cluster_id = row["cluster"]
        cluster_name = row["cluster_name"]
        count = int(row.get("count", 0))
        pct = float(row.get("pct", 0.0))

        df_topic = df_clustered[df_clustered["cluster"] == cluster_id]
        df_recent = _recent_topic_window(df_topic, max_days=30, min_rows=12)
        df_recent = df_recent.sort_values("published_at", ascending=False)
        context_headlines = _pick_headlines_from_rows(df_recent, n=8)
        context_blob = " ".join([cluster_name] + context_headlines).lower()

        event_hits = sum(1 for term in event_terms if term.lower() in context_blob)
        inertia_count = int(inertia_counts.get(cluster_id, 0))
        outlet_profile_score, prefer_hits, avoid_hits = _outlet_profile_signal(context_blob, outlet)

        freq_score = max(0.0, 8.0 - float(rank - 1)) + pct / 25.0
        inertia_score = min(inertia_count, 4) * 1.5
        event_score = min(event_hits, 4) * 1.25
        ensemble_score = freq_score + inertia_score + event_score + outlet_profile_score

        reasons = [f"частотная база {count} новостей ({pct:.1f}%)"]
        if inertia_count:
            reasons.append(f"тема держится со вчера ({inertia_count} публикаций)")
        if event_hits:
            reasons.append(f"есть календарная поддержка ({event_hits} пересечений)")
        if prefer_hits:
            reasons.append("совпадает с профилем издания: " + ", ".join(prefer_hits[:4]))
        if avoid_hits:
            reasons.append("штраф за чуждый угол: " + ", ".join(avoid_hits[:3]))

        hybrid_info.append({
            "cluster": cluster_id,
            "cluster_name": cluster_name,
            "name": cluster_name,
            "count": count,
            "pct": pct,
            "freq_rank": rank,
            "freq_score": float(freq_score),
            "inertia_count": inertia_count,
            "event_hits": int(event_hits),
            "event_score": float(event_score),
            "outlet_profile_score": float(outlet_profile_score),
            "outlet_prefer_hits": prefer_hits,
            "outlet_avoid_hits": avoid_hits,
            "forward_signal_score": 0.0,
            "forward_signal_count": 0,
            "forward_target_hits": 0,
            "forward_examples": [],
            "forward_markers": [],
            "ensemble_score": float(ensemble_score),
            "selection_reasons": reasons,
            "topic_strategy": "hybrid",
            "forecast_profile": "default",
        })

    hybrid_info.sort(
        key=lambda item: (
            float(item.get("ensemble_score", 0.0)),
            float(item.get("outlet_profile_score", 0.0)),
            int(item.get("inertia_count", 0)),
            int(item.get("event_hits", 0)),
            float(item.get("pct", 0.0)),
            -int(item.get("freq_rank", 999)),
        ),
        reverse=True,
    )
    hybrid_info = hybrid_info[:top_n]
    if profile == "forward_look":
        return _apply_forward_look_profile(hybrid_info, forward_stats, top_n=top_n)
    return hybrid_info


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

_OUTLET_PROFILE_PRIORS = {
    "kommersant": {
        "prefer_weight": 1.45,
        "avoid_weight": 2.15,
        "prefer": [
            "нефт", "санкц", "логист", "маршрут", "постав", "экспорт", "импорт",
            "рынк", "рубл", "банк", "цб", "центробанк", "компан", "налог",
            "кодекс", "груз", "танкер", "страхов", "нефтепродукт",
        ],
        "avoid": [
            "чемпионат", "фигур", "матч", "золото", "тренер", "футбол",
            "турнир", "сборн", "минобороны", "взятии", "боеприпас",
            "удар по", "всу атак", "спорт",
        ],
        "hard_avoid": [
            "минобороны", "взятии", "уничтожении", "боеприпас", "днр",
        ],
    },
    "lenta": {
        "prefer_weight": 1.45,
        "avoid_weight": 1.55,
        "prefer": [
            "удар", "атак", "бпла", "дрон", "всу", "взрыв", "пожар",
            "эваку", "пригроз", "учени", "пво", "обстрел", "разруш",
        ],
        "avoid": [
            "дивиденд", "совет директоров", "налоговый кодекс", "центробанк",
            "акционер", "бирж", "рсбу", "комитет рассмотрит", "отчетность",
        ],
    },
    "interfax": {
        "prefer_weight": 1.4,
        "avoid_weight": 1.65,
        "prefer": [
            "заяв", "подтверд", "обсуд", "переговор", "санкц", "оцен",
            "маршрут", "постав", "рынк", "комисси", "министер", "госдеп",
            "нато", "ек ", "еврокомисс", "разведк", "коридор", "делегац",
        ],
        "avoid": [
            "чемпионат", "фигур", "золото", "тренер", "болельщик", "пожар на складе",
            "губернатор сообщил", "жилой дом", "звезда",
        ],
    },
}

_OUTLET_GUIDANCE = {
    "kommersant": (
        "Делай упор на бизнес-угол: санкции, нефть, поставки, логистику, компании, "
        "регуляторов и экономические последствия. Избегай спортивных и чисто фронтовых сводок."
    ),
    "lenta": (
        "Делай упор на конфликтные и ударные сюжеты: атаки, БПЛА, эвакуации, угрозы, "
        "силовые эпизоды и резкие заявления."
    ),
    "interfax": (
        "Делай упор на агентский и официально-дипломатический угол: заявления ведомств, "
        "переговоры, санкции, оценки рынков, логистику и международные консультации."
    ),
}

_KOMMERSANT_MILITARY_MARKERS = [
    "минобороны", "всу", "удар", "взятии", "взятии", "уничтожении",
    "боеприпас", "наступлен", "штурм", "дрон", "бпла", "днр",
]

_KOMMERSANT_BUSINESS_POLICY_MARKERS = [
    "санкц", "нефт", "газ", "рынк", "постав", "экспорт", "импорт",
    "логист", "маршрут", "компан", "бизнес", "цб", "центробанк",
    "госдума", "правитель", "пошлин", "страхов", "резерв", "рубл",
    "переговор", "делегац", "регулятор", "кодекс",
]

_LENTA_CONFLICT_MARKERS = [
    "удар", "атак", "бпла", "дрон", "всу", "взрыв", "обстрел", "эваку",
    "пригроз", "пво", "пожар", "разруш", "поврежден", "учени", "учения",
    "боевых действий", "войн", "конфликт", "угроз",
]

_LENTA_BUREAUCRATIC_MARKERS = [
    "совет директоров", "дивиденд", "центробанк", "цб", "налоговый кодекс",
    "комитет рассмотрит", "отчетность", "рсбу", "бирж", "гост",
    "регулятор", "акционер", "котировк", "экспортная нефть",
]

_INTERFAX_OFFICIAL_MARKERS = [
    "заяв", "подтверд", "обсуд", "переговор", "санкц", "оцен",
    "комисси", "госдеп", "нато", "еврокомисс", "ек ", "министер",
    "совбез", "мид", "делегац", "коридор", "маршрут", "рынк",
    "постав", "разведк", "предложил", "сообщил", "объявил",
]

_INTERFAX_SENSATIONAL_MARKERS = [
    "губернатор сообщил", "жилой дом", "пожар", "склад", "пострад",
    "разруш", "дрон", "бпла", "звезда", "болельщик", "шок", "паник",
]


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


def _subcluster_topic_rows(
    parent_label: str,
    df_topic: pd.DataFrame,
    max_subtopics: int = 2,
    min_rows_for_split: int = 30,
) -> List[Dict]:
    """Split a broad topic into a few tighter subtopics before LLM generation."""
    ranked = df_topic.sort_values("published_at", ascending=False).copy()
    texts = _texts_for_outlet(ranked)
    unique_texts = {text.strip() for text in texts if text.strip()}

    if len(ranked) < min_rows_for_split or len(unique_texts) < 3:
        return [{
            "label": parent_label,
            "rows": ranked,
            "subtopic_size": int(len(ranked)),
            "subtopic_rank": 0,
        }]

    stopwords = _get_stopwords() | _GROUNDING_NOISE_WORDS
    min_df = 2 if len(ranked) >= 40 else 1
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[а-яА-ЯёЁa-zA-Z]{3,}\b",
        min_df=min_df,
        max_df=0.75,
        max_features=2500,
        stop_words=list(stopwords),
    )
    try:
        matrix = vec.fit_transform(texts)
    except ValueError:
        return [{
            "label": parent_label,
            "rows": ranked,
            "subtopic_size": int(len(ranked)),
            "subtopic_rank": 0,
        }]

    n_clusters = 2 if len(ranked) < 100 else 3
    n_clusters = min(max_subtopics, n_clusters, max(1, len(unique_texts) - 1), matrix.shape[0] - 1)
    if n_clusters < 2:
        return [{
            "label": parent_label,
            "rows": ranked,
            "subtopic_size": int(len(ranked)),
            "subtopic_rank": 0,
        }]

    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(matrix)
        terms = vec.get_feature_names_out()
    except Exception:
        return [{
            "label": parent_label,
            "rows": ranked,
            "subtopic_size": int(len(ranked)),
            "subtopic_rank": 0,
        }]

    subtopics: List[Dict] = []
    counts = Counter(labels.tolist())
    for rank, cluster_id in enumerate(sorted(counts, key=counts.get, reverse=True)):
        rows = ranked.iloc[np.where(labels == cluster_id)[0]].copy()
        center = km.cluster_centers_[cluster_id]
        top_idx = center.argsort()[-4:][::-1]
        sub_name = " / ".join(
            term for term in (terms[i] for i in top_idx)
            if term and term.strip()
        )
        label = parent_label if not sub_name else f"{parent_label} | {sub_name}"
        subtopics.append({
            "label": label,
            "rows": rows.sort_values("published_at", ascending=False),
            "subtopic_size": int(len(rows)),
            "subtopic_rank": rank,
        })

    return subtopics[:max_subtopics] or [{
        "label": parent_label,
        "rows": ranked,
        "subtopic_size": int(len(ranked)),
        "subtopic_rank": 0,
    }]


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
    outlet_guidance: str = "",
    forward_context: str = "",
    requested_items: int = 4,
) -> str:
    outlet_name  = OUTLETS[outlet]["name"]
    date_str     = target_date.strftime("%d.%m.%Y")
    examples_str = "\n\n".join(style_examples) if style_examples else "(примеры недоступны)"
    entities_str = ", ".join(top_entities[:10]) if top_entities else "нет данных"
    keywords_str = ", ".join(topic_keywords[:10]) if topic_keywords else "нет данных"
    headlines_str = "\n".join(f"- {h}" for h in recent_headlines[:6]) if recent_headlines else "(связанные заголовки недоступны)"
    forward_block = (
        "Явные анонсы и упоминания целевой даты в недавних публикациях:\n"
        f"{forward_context}"
        if forward_context else
        ""
    )

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
{forward_block}

# OBJECTIVE
Сгенерируй {requested_items} правдоподобных новостных сюжета на указанную дату, которые могли бы естественным образом появиться в издании «{outlet_name}».
Используй актуальную повестку (календарные события и реальных действующих лиц) вместо абстрактных или выдуманных (фантастических) событий.
Каждый сюжет должен быть явно связан хотя бы с одной сущностью или ключевым словом из списка выше.
Предпочитай развитие уже наблюдаемой повестки, а не случайные новые ветки.
Не смешивай несколько разных инфоповодов: держись одной конкретной подтемы, которая лучше всего подтверждается заголовками и ключевыми словами из контекста.
{"Редакционный угол для этого издания: " + outlet_guidance if outlet_guidance else ""}

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
    signature_terms: Optional[List[str]] = None,
    topic_weight: float = 0.0,
    outlet: str = "",
    outlet_profile_weight: float = 0.0,
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
    signature_set = {
        term.lower() for term in (signature_terms or [])
        if isinstance(term, str) and term.strip()
    }

    scored = []
    for item in results:
        title = str(item.get("title") or "").strip()
        lead = str(item.get("lead") or "").strip()
        text = f"{title} {lead}".lower()

        keyword_hits = sum(1 for kw in keyword_set if kw in text)
        entity_hits = sum(1 for ent in entity_set if ent in text)
        recent_hits = sum(1 for term in recent_term_set if term in text)
        signature_hits = sum(1 for term in signature_set if term in text)
        outlet_profile_score, prefer_hits, avoid_hits = _outlet_profile_signal(text, outlet)
        story_gate_score, gate_hits, context_hits = _outlet_story_gate(text, outlet)
        title_len = len(title.split())
        score = (
            topic_weight
            + entity_hits * 4
            + keyword_hits * 3
            + recent_hits * 2
            + signature_hits * 2
            + outlet_profile_score * outlet_profile_weight
            + story_gate_score
        )
        if outlet_profile_weight > 0 and avoid_hits:
            score -= len(avoid_hits) * (0.75 + outlet_profile_weight * 0.5)
        if 6 <= title_len <= 14:
            score += 1.0
        if lead:
            score += min(len(lead.split()), 40) / 40
        if "?" in title or "!" in title:
            score -= 1.0

        enriched = dict(item)
        enriched["_llm_score"] = float(score)
        enriched["_outlet_profile_score"] = float(outlet_profile_score)
        enriched["_outlet_prefer_hits"] = prefer_hits
        enriched["_outlet_avoid_hits"] = avoid_hits
        enriched["_story_gate_score"] = float(story_gate_score)
        enriched["_story_gate_hits"] = gate_hits
        enriched["_story_gate_context_hits"] = context_hits
        scored.append((score, enriched))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [item for score, item in scored if score > 0]
    if len(filtered) < min_keep:
        filtered = [item for _, item in scored[:max(min_keep, len(filtered))]]

    seen_titles = []
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


def _normalized_title_key(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", title.lower())).strip()


def _rerank_llm_results(results: List[Dict], keep_n: int = 10,
                        per_parent_limit: int = 4) -> List[Dict]:
    """Global reranking across subtopics, with light diversity control."""
    if not results:
        return []

    seen_keys = set()
    per_parent_counts: Counter = Counter()
    ranked: List[Dict] = []

    sorted_results = sorted(
        results,
        key=lambda item: (
            float(item.get("_llm_score", 0.0)),
            float(item.get("_topic_priority", 0.0)),
            len(str(item.get("lead") or "").split()),
        ),
        reverse=True,
    )

    for item in sorted_results:
        title = str(item.get("title") or "").strip()
        if not title:
            continue

        title_key = _normalized_title_key(title)
        if not title_key or title_key in seen_keys:
            continue

        parent_key = str(item.get("_topic_parent", ""))
        if per_parent_counts[parent_key] >= per_parent_limit:
            continue

        seen_keys.add(title_key)
        per_parent_counts[parent_key] += 1
        ranked.append({
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        })
        if len(ranked) >= keep_n:
            break

    return ranked


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
    
    candidate_pool: List[Dict] = []
    total_topics = max(1, min(n_topics, len(topic_info)))
    for topic_rank, info in enumerate(topic_info[:n_topics]):
        label = info["name"]
        cluster_id = info["cluster"]

        # Context strictly isolated to current topic
        df_topic = df_clustered[df_clustered["cluster"] == cluster_id]
        df_topic_recent = _recent_topic_window(df_topic, max_days=30, min_rows=20)
        subtopics = _subcluster_topic_rows(label, df_topic_recent, max_subtopics=2)

        for sub in subtopics:
            sub_label = sub["label"]
            sub_rows = sub["rows"]

            signature_terms = _topic_signature_terms_from_rows(sub_rows, topic_vec, top_k=10)
            support_rows = _retrieve_topic_rows(sub_rows, signature_terms, topic_vec, top_k=8)
            support_headlines = _pick_headlines_from_rows(support_rows, n=8)

            phrase_entities = _headline_phrase_candidates(support_headlines, top_k=8)
            if phrase_entities:
                top_entities = phrase_entities
            else:
                topic_entities_data = entity_frequency(support_rows if not support_rows.empty else sub_rows)
                top_persons = topic_entities_data.get("persons", pd.Series()).index.tolist()[:7]
                top_orgs = topic_entities_data.get("orgs", pd.Series()).index.tolist()[:7]
                raw_entities = top_persons + top_orgs
                top_entities = _clean_grounding_entities(raw_entities, support_headlines, top_k=8)

            topic_keywords = _topic_keywords(sub_label, top_entities + signature_terms, support_headlines)

            style_examples = _pick_style_examples_from_rows(support_rows, n=4)
            if len(style_examples) < 6:
                style_examples.extend(_pick_style_examples(df, n=6 - len(style_examples)))

            topic_priority = (total_topics - topic_rank) * 1.5 + float(info.get("pct", 0.0)) / 50.0
            if info.get("topic_strategy") == "hybrid":
                topic_priority += float(info.get("ensemble_score", 0.0)) / 4.0
            if info.get("forecast_profile") == "forward_look":
                topic_priority += float(info.get("forward_signal_score", 0.0)) / 4.0
            subtopic_bonus = max(0.0, 1.0 - float(sub.get("subtopic_rank", 0)) * 0.3)
            topic_weight = topic_priority + subtopic_bonus
            outlet_guidance = _OUTLET_GUIDANCE.get(outlet, "") if info.get("topic_strategy") == "hybrid" else ""
            forward_examples = list(info.get("forward_examples", []))
            forward_context = "\n".join(f"- {title}" for title in forward_examples[:3])

            metrics_parts = [(
                f"Популярность темы в последнее время: {info['count']} новостей "
                f"({info['pct']}% от общего потока). "
                f"Размер текущей подтемы: {sub.get('subtopic_size', len(sub_rows))} материалов."
            )]
            if info.get("topic_strategy") == "hybrid":
                metrics_parts.append(
                    "Сигналы отбора: "
                    f"frequency={float(info.get('freq_score', 0.0)):.2f}, "
                    f"inertia={int(info.get('inertia_count', 0))}, "
                    f"calendar={int(info.get('event_hits', 0))}, "
                    f"outlet={float(info.get('outlet_profile_score', 0.0)):.2f}, "
                    f"ensemble={float(info.get('ensemble_score', 0.0)):.2f}."
                )
                reasons = info.get("selection_reasons") or []
                if reasons:
                    metrics_parts.append("Причины выбора темы: " + "; ".join(reasons) + ".")
            if info.get("forecast_profile") == "forward_look":
                metrics_parts.append(
                    "Forward-looking сигналы: "
                    f"score={float(info.get('forward_signal_score', 0.0)):.2f}, "
                    f"count={int(info.get('forward_signal_count', 0))}, "
                    f"direct_target_hits={int(info.get('forward_target_hits', 0))}."
                )
            metrics = " ".join(metrics_parts)

            requested_items = 3 if len(support_rows) >= 6 else 2
            min_keep = 2 if requested_items == 2 else 3

            print(f"  [llm] {outlet} | topic: {sub_label[:60]}...")
            prompt = _build_llm_prompt(
                outlet, target_date, date_meta, sub_label, metrics,
                events_summary, style_examples, support_headlines, top_entities,
                topic_keywords, has_lead=has_lead, outlet_guidance=outlet_guidance,
                forward_context=forward_context,
                requested_items=requested_items,
            )
            try:
                raw = _llm_chat(prompt)
                parsed = _parse_llm_response(raw, outlet, target_date, sub_label)
                filtered = _filter_llm_results(
                    parsed,
                    topic_keywords,
                    top_entities,
                    recent_headlines=support_headlines,
                    signature_terms=signature_terms,
                    topic_weight=topic_weight,
                    outlet=outlet,
                    outlet_profile_weight=1.25 if info.get("topic_strategy") == "hybrid" else 0.0,
                    min_keep=min_keep,
                )
                for item in filtered:
                    enriched = dict(item)
                    enriched["_topic_parent"] = str(cluster_id)
                    enriched["_topic_priority"] = float(topic_priority)
                    candidate_pool.append(enriched)
            except Exception as exc:
                print(f"  [llm] Error for {outlet}/{sub_label}: {exc}")

    keep_n = max(6, total_topics * 4)
    return _rerank_llm_results(candidate_pool, keep_n=keep_n, per_parent_limit=4)


# ================================================================
#  COMBINE FORECAST
# ================================================================

def combine_forecast(
    outlet: str,
    target_date: datetime.date = TARGET_DATE,
    use_llm: bool = True,
    llm_only: bool = True,
    forecast_strategy: str = "llm",
    forecast_profile: str = "default",
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
    frequency_top_topics = freq["cluster_name"].head(5).tolist()

    # Events context
    events = get_events_for_outlet(outlet, target_date, window_days=3)
    events_summary = summarize_events(events)

    # Prepare topic info for LLM
    topic_info = _prepare_llm_topic_info(
        df,
        labels,
        freq,
        outlet,
        target_date,
        events,
        strategy=forecast_strategy,
        profile=forecast_profile,
        top_n=5,
    )
    top_topics = [info.get("name", "unknown") for info in topic_info[:5]] or frequency_top_topics

    # LLM
    llm_results: List[Dict] = []
    if use_llm:
        llm_results = llm_forecast(df, labels, outlet, target_date,
                                   topic_info, events_summary, events)

    if llm_only:
        all_preds = llm_results
        inertia = []
        freq_fc = []
        calendar = []
    else:
        inertia = inertia_forecast(df, outlet, target_date)
        freq_fc = frequency_forecast(df, outlet, target_date)
        calendar = calendar_forecast(outlet, target_date)
        all_preds = inertia + freq_fc + calendar + llm_results

    report = {
        "outlet":           outlet,
        "outlet_name":      outlet_name,
        "target_date":      str(target_date),
        "forecast_strategy": forecast_strategy,
        "forecast_profile": forecast_profile,
        "top_topics":       top_topics,
        "frequency_top_topics": frequency_top_topics,
        "topic_selection":  [
            {
                "topic_label": info.get("name", "unknown"),
                "count": int(info.get("count", 0)),
                "pct": float(info.get("pct", 0.0)),
                "freq_score": float(info.get("freq_score", 0.0)),
                "inertia_count": int(info.get("inertia_count", 0)),
                "event_hits": int(info.get("event_hits", 0)),
                "outlet_profile_score": float(info.get("outlet_profile_score", 0.0)),
                "outlet_prefer_hits": list(info.get("outlet_prefer_hits", [])),
                "outlet_avoid_hits": list(info.get("outlet_avoid_hits", [])),
                "forward_signal_score": float(info.get("forward_signal_score", 0.0)),
                "forward_signal_count": int(info.get("forward_signal_count", 0)),
                "forward_target_hits": int(info.get("forward_target_hits", 0)),
                "forward_examples": list(info.get("forward_examples", [])),
                "forward_markers": list(info.get("forward_markers", [])),
                "ensemble_score": float(info.get("ensemble_score", 0.0)),
                "selection_reasons": list(info.get("selection_reasons", [])),
            }
            for info in topic_info[:5]
        ],
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
    llm_only: bool = True,
    forecast_strategy: str = "llm",
    forecast_profile: str = "default",
) -> Dict[str, Dict]:
    """
    Forecast for all outlets. Saves JSON to data/forecasts/.
    """
    from config import OUTLET_SLUGS
    if slugs is None:
        slugs = OUTLET_SLUGS

    run_started_at = datetime.datetime.now()
    run_stamp = _forecast_run_stamp(run_started_at)
    generated_at = run_started_at.isoformat(timespec="seconds")
    all_reports: Dict[str, Dict] = {}
    for slug in slugs:
        report = combine_forecast(
            slug,
            target_date,
            use_llm=use_llm,
            llm_only=llm_only,
            forecast_strategy=forecast_strategy,
            forecast_profile=forecast_profile,
        )
        report["generated_at"] = generated_at
        report["forecast_run_id"] = run_stamp
        all_reports[slug] = report

    # Write combined JSON
    out_path, xlsx_path = _forecast_output_paths(
        target_date,
        run_stamp,
        forecast_strategy,
        forecast_profile,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)

    export_rows = []
    for slug, report in all_reports.items():
        outlet_name = report.get("outlet_name", slug)
        top_topics = " | ".join(report.get("top_topics", [])[:5])
        for pred in report.get("predictions", []):
            export_rows.append({
                "outlet": slug,
                "outlet_name": outlet_name,
                "target_date": report.get("target_date"),
                "forecast_strategy": report.get("forecast_strategy"),
                "forecast_profile": report.get("forecast_profile"),
                "method": pred.get("method"),
                "rubric": pred.get("rubric"),
                "topic_label": pred.get("topic_label"),
                "title": pred.get("title"),
                "lead": pred.get("lead"),
                "top_topics": top_topics,
            })

    df_export = pd.DataFrame(export_rows)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="Все прогнозы", index=False)

        df_llm = df_export[df_export["method"] == "llm"]
        if not df_llm.empty:
            df_llm.to_excel(writer, sheet_name="LLM заголовки", index=False)

        for slug, report in all_reports.items():
            rows = [row for row in export_rows if row["outlet"] == slug]
            if not rows:
                continue
            pd.DataFrame(rows).to_excel(writer, sheet_name=slug[:31], index=False)

    print(f"\n[forecast] Saved → {out_path}")
    print(f"[forecast] Saved → {xlsx_path}")
    return all_reports
