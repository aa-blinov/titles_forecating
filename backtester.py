"""
Backtester: holdout evaluation on the last BACKTEST_DAYS days of clean data.
For each day in holdout, trains on everything before that day,
generates forecasts, stores predictions vs actuals for metrics.
"""
import datetime
import json
import os
from typing import Dict, List, Optional

import pandas as pd

from config import OUTLET_SLUGS, BACKTEST_DAYS, BACKTESTS_DIR, FREQ_WINDOW
from etl import load_clean
from forecaster import (
    inertia_forecast, frequency_forecast, calendar_forecast, llm_forecast,
)
from analyzer import extract_topics, topic_frequency
from event_calendar import get_events_for_outlet, summarize_events


# ================================================================
#  SINGLE-DAY BACKTEST
# ================================================================

def backtest_day(
    df_full: pd.DataFrame,
    outlet: str,
    target_date: datetime.date,
    methods: List[str],
) -> Dict:
    """
    Given full dataframe, split at target_date and run forecasts
    on the training portion only.
    Returns dict with predicted topics/titles and actual titles for that day.
    """
    cutoff    = pd.Timestamp(target_date)
    df_train  = df_full[df_full["published_at"] < cutoff].copy()
    df_actual = df_full[df_full["published_at"].dt.date == target_date].copy()

    if df_train.empty:
        return {}

    day_result: Dict = {
        "outlet":  outlet,
        "date":    str(target_date),
        "actual":  {
            "titles":  df_actual["title"].dropna().tolist(),
            "rubrics": df_actual["rubric"].dropna().tolist(),
        },
        "predictions": {},
    }

    # Pre-calculate shared context if LLM is requested
    llm_context = None
    if "llm" in methods:
        labels, names, _ = extract_topics(df_train, outlet)
        freq = topic_frequency(df_train, labels, names)
        topic_info = freq.head(5).to_dict("records")
        for ti in topic_info: ti["name"] = ti.get("cluster_name", "unknown")
        
        events = get_events_for_outlet(outlet, target_date, window_days=2)
        events_summary = summarize_events(events)
        llm_context = (labels, topic_info, events_summary, events)

    for method in methods:
        try:
            if method == "inertia":
                preds = inertia_forecast(df_train, outlet, target_date)
            elif method == "frequency":
                preds = frequency_forecast(df_train, outlet, target_date)
            elif method == "calendar":
                preds = calendar_forecast(outlet, target_date)
            elif method == "llm":
                if llm_context:
                    labs, ti, es, el = llm_context
                    # Limit LLM to 2 topics in backtest to save time/tokens
                    preds = llm_forecast(df_train, labs, outlet, target_date, ti, es, el, n_topics=2)
                else:
                    preds = []
            else:
                continue
                
            day_result["predictions"][method] = [
                {k: v for k, v in p.items()
                 if k in ("rubric", "topic_label", "title", "lead", "method")}
                for p in preds
            ]
        except Exception as exc:
            print(f"  [backtest] Error method={method} date={target_date}: {exc}")
            day_result["predictions"][method] = []

    return day_result


# ================================================================
#  FULL HOLDOUT BACKTEST FOR ONE OUTLET
# ================================================================

def run_backtest(
    outlet: str,
    holdout_days: int = BACKTEST_DAYS,
    methods: List[str] = None,
) -> List[Dict]:
    """
    Run backtest for one outlet over the last holdout_days.
    Returns list of per-day result dicts.
    """
    if methods is None:
        methods = ["inertia", "frequency", "calendar", "llm"]

    df = load_clean(outlet)
    if df.empty:
        print(f"  [backtest] No clean data for {outlet}")
        return []

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"]).sort_values("published_at")

    max_date = df["published_at"].max().date()
    holdout_dates = [
        max_date - datetime.timedelta(days=i)
        for i in range(holdout_days - 1, -1, -1)
    ]

    def _history_days_before(day: datetime.date) -> int:
        return int(df[df["published_at"].dt.date < day]["published_at"].dt.date.nunique())

    valid_holdout_dates = []
    skipped_days = []
    for day in holdout_dates:
        history_days = _history_days_before(day)
        if history_days < FREQ_WINDOW:
            skipped_days.append((day, history_days))
            continue
        valid_holdout_dates.append(day)

    if skipped_days:
        print(
            f"  [backtest] skipped {len(skipped_days)} day(s) for {outlet} "
            f"with < {FREQ_WINDOW} history days"
        )

    if not valid_holdout_dates:
        print(
            f"  [backtest] {outlet}: no valid holdout dates "
            f"(need at least {FREQ_WINDOW} history days)"
        )
        return []

    print(f"\n[backtest] {outlet}: holdout {valid_holdout_dates[0]} → {valid_holdout_dates[-1]}")

    results: List[Dict] = []
    for day in valid_holdout_dates:
        day_res = backtest_day(df, outlet, day, methods)
        if day_res:
            results.append(day_res)
            actual_n = len(day_res["actual"]["titles"])
            print(f"  {day}: {actual_n} actual articles")

    return results


# ================================================================
#  SAVE / LOAD BACKTEST RESULTS
# ================================================================

def save_backtest(outlet: str, results: List[Dict]) -> str:
    today_str = datetime.date.today().strftime("%Y%m%d")
    path = os.path.join(BACKTESTS_DIR, f"backtest_{outlet}_{today_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [backtest] saved → {path}")
    return path


def load_backtest(outlet: str) -> List[Dict]:
    """Load the most recent backtest file for an outlet."""
    if not os.path.exists(BACKTESTS_DIR):
        return []
    files = sorted([
        f for f in os.listdir(BACKTESTS_DIR)
        if f.startswith(f"backtest_{outlet}_") and f.endswith(".json")
    ])
    if not files:
        return []
    path = os.path.join(BACKTESTS_DIR, files[-1])
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ================================================================
#  RUN ALL
# ================================================================

def backtest_all(
    slugs: List[str] = None,
    holdout_days: int = BACKTEST_DAYS,
    methods: List[str] = None,
) -> Dict[str, List[Dict]]:
    if slugs is None:
        slugs = OUTLET_SLUGS
    if methods is None:
        methods = ["inertia", "frequency", "calendar", "llm"]

    all_results: Dict[str, List[Dict]] = {}
    for slug in slugs:
        res = run_backtest(slug, holdout_days, methods)
        save_backtest(slug, res)
        all_results[slug] = res

    return all_results
