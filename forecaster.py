"""
Forecaster:
  1. Inertia baseline   — yesterday's topic distribution
  2. Frequency baseline — top topics last 14 days
  3. Calendar baseline  — titles from known events
  4. LLM generation     — Ollama llama3.1 in outlet's style
"""
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import (
    OUTLETS, FORECASTS_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, OPENROUTER_MODEL,
    FREQ_WINDOW, TARGET_DATE,
)
from etl import load_clean
from analyzer import (
    extract_topics, topic_frequency, entity_frequency,
    style_profile, _texts_for_outlet,
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
    )
    
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.75,
        max_tokens=800,
        extra_headers={
            "HTTP-Referer": "https://github.com/aazhivotrev/titles_forecating", # Optional
            "X-Title": "Titles Forecasting Project", # Optional
        }
    )
    return response.choices[0].message.content.strip()


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

def calendar_forecast(outlet: str, target_date: datetime.date) -> List[Dict]:
    """
    Template headlines from known calendar events.
    """
    outlet_name = OUTLETS[outlet]["name"]
    events = get_events_for_outlet(outlet, target_date, window_days=1)
    results = []
    for ev in events:
        # Build a template headline
        rubric = ev["event_type"]
        desc   = ev["description"]
        title  = f"[Шаблон] {outlet_name}: {desc}"
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


# ================================================================
#  LLM — PROMPT BUILDER
# ================================================================

def _build_llm_prompt(
    outlet: str,
    target_date: datetime.date,
    topic_label: str,
    events_summary: str,
    style_examples: List[str],
    top_entities: List[str],
    has_lead: bool,
) -> str:
    outlet_name  = OUTLETS[outlet]["name"]
    date_str     = target_date.strftime("%d %B %Y")
    examples_str = "\n\n".join(style_examples) if style_examples else "(примеры недоступны)"
    entities_str = ", ".join(top_entities[:10]) if top_entities else "нет данных"

    lead_instruction = (
        "Для каждой новости напиши: ЗАГОЛОВОК и ЛИД (1-2 предложения, до 50 слов)."
        if has_lead else
        "Напиши только ЗАГОЛОВОК для каждой новости."
    )

    return f"""Ты — опытный редактор издания «{outlet_name}».
Дата прогноза: {date_str}.
Тематическая область: «{topic_label}».

Известные события на эту дату и рядом:
{events_summary}

Часто упоминаемые персоны и организации в последние недели:
{entities_str}

Примеры заголовков (и лидов) в стиле «{outlet_name}»:
{examples_str}

Задача: напиши 5 правдоподобных новостных заголовков, которые могло бы опубликовать издание «{outlet_name}» {date_str}. Опирайся на тематику, стиль примеров и известные события. НЕ копируй примеры дословно. Строго придерживайся tone-of-voice издания.

{lead_instruction}

Формат ответа (строго):
1. ЗАГОЛОВОК: ...
   ЛИД: ...
2. ЗАГОЛОВОК: ...
   ЛИД: ...
...
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


# ================================================================
#  LLM — MAIN GEN FUNCTION
# ================================================================

def llm_forecast(
    df: pd.DataFrame,
    outlet: str,
    target_date: datetime.date,
    topic_labels: List[str],
    events_summary: str,
    n_topics: int = 3,
) -> List[Dict]:
    """
    Generate LLM-based forecasts for top N topics.
    """
    if not check_llm_availability():
        print(f"  [llm] Skipping LLM for {outlet} — API key missing")
        return []

    has_lead = OUTLETS[outlet]["has_lead"]
    top_entities_data = entity_frequency(df)
    top_persons  = top_entities_data.get("persons", pd.Series()).index.tolist()[:10]
    top_orgs     = top_entities_data.get("orgs", pd.Series()).index.tolist()[:10]
    top_entities = top_persons + top_orgs

    style_examples = _pick_style_examples(df, n=5)

    all_results = []
    for topic in topic_labels[:n_topics]:
        print(f"  [llm] {outlet} | topic: {topic[:60]}...")
        prompt = _build_llm_prompt(
            outlet, target_date, topic,
            events_summary, style_examples, top_entities,
            has_lead=has_lead,
        )
        try:
            raw = _llm_chat(prompt)
            parsed = _parse_llm_response(raw, outlet, target_date, topic)
            all_results.extend(parsed)
        except Exception as exc:
            print(f"  [llm] Error for {outlet}/{topic}: {exc}")

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
    events = get_events_for_outlet(outlet, target_date, window_days=2)
    events_summary = summarize_events(events)

    # Baselines
    inertia  = inertia_forecast(df, outlet, target_date)
    freq_fc  = frequency_forecast(df, outlet, target_date)
    calendar = calendar_forecast(outlet, target_date)

    # LLM
    llm_results: List[Dict] = []
    if use_llm:
        llm_results = llm_forecast(df, outlet, target_date,
                                   top_topics, events_summary)

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
