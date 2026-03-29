"""
Event calendar loader: reads events.csv and filters by date + outlet.
"""
import datetime
import os
from typing import List, Dict, Optional

import pandas as pd

from config import EVENTS_CSV, OUTLETS


def _build_event(date: datetime.date, event_type: str, description: str,
                 outlets: List[str], source: str) -> Dict:
    return {
        "date": date,
        "event_type": event_type,
        "description": description,
        "outlets": outlets,
        "source": source,
    }


def _daterange(lo: datetime.date, hi: datetime.date):
    current = lo
    while current <= hi:
        yield current
        current += datetime.timedelta(days=1)


def _recurring_events(lo: datetime.date, hi: datetime.date) -> List[Dict]:
    """
    Lightweight fallback layer for predictable weekly/month-edge agenda.
    Keeps calendar forecasts from going empty when events.csv is sparse.
    """
    events: List[Dict] = []
    for day in _daterange(lo, hi):
        if day.weekday() in (1, 2, 3):
            events.append(_build_event(
                day,
                "politics",
                "Плановая парламентская неделя: заседания Госдумы и профильных комитетов",
                ["kommersant", "lenta", "interfax", "rbc"],
                "recurring_parliament",
            ))

        if day.weekday() == 3:
            events.append(_build_event(
                day,
                "economics",
                "Еженедельная статистика рынка труда США",
                ["kommersant", "interfax", "vedomosti", "rbc", "lenta"],
                "recurring_us_labor",
            ))

        if day.weekday() in (5, 6):
            events.append(_build_event(
                day,
                "sport",
                "Матчи выходного дня: РПЛ, КХЛ и другие крупные турниры",
                ["kommersant", "lenta", "interfax", "rbc"],
                "recurring_sports",
            ))

        if day.day >= 28 or day.day <= 3:
            events.append(_build_event(
                day,
                "economics",
                "Конец месяца и новые оперативные данные по экономике и компаниям",
                ["kommersant", "interfax", "vedomosti", "rbc"],
                "recurring_month_edge",
            ))
    return events


def load_events(
    target_date: Optional[datetime.date] = None,
    outlet: Optional[str] = None,
    window_days: int = 3,
) -> List[Dict]:
    """
    Load events from events.csv, optionally filtered by date window and outlet.

    Parameters
    ----------
    target_date : date to centre the window on (default: all events)
    outlet      : outlet slug to filter by (default: all outlets)
    window_days : ±days around target_date to include

    Returns
    -------
    List of event dicts: date, event_type, description, outlets, source
    """
    if not os.path.exists(EVENTS_CSV):
        print(f"[calendar] events.csv not found: {EVENTS_CSV}")
        return []

    df = pd.read_csv(EVENTS_CSV, dtype=str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    if target_date is not None:
        lo = target_date - datetime.timedelta(days=window_days)
        hi = target_date + datetime.timedelta(days=window_days)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    else:
        valid_dates = df["date"].dropna()
        if valid_dates.empty:
            return []
        lo = valid_dates.min()
        hi = valid_dates.max()

    events: List[Dict] = []
    for _, row in df.iterrows():
        events.append(_build_event(
            row["date"],
            row.get("event_type", ""),
            row.get("description", ""),
            [o.strip() for o in str(row.get("outlets", "")).split(";") if o.strip()],
            row.get("source", ""),
        ))

    events.extend(_recurring_events(lo, hi))

    if outlet is not None:
        events = [ev for ev in events if outlet in ev.get("outlets", [])]

    deduped = {}
    for ev in events:
        key = (
            ev.get("date"),
            ev.get("event_type"),
            ev.get("description"),
            tuple(sorted(ev.get("outlets", []))),
        )
        deduped[key] = ev

    return sorted(deduped.values(), key=lambda ev: (ev.get("date") or datetime.date.min, ev.get("event_type", ""), ev.get("description", "")))


def get_events_for_outlet(outlet: str, target_date: datetime.date,
                          window_days: int = 1) -> List[Dict]:
    """Convenience: events for a specific outlet on/around target_date."""
    return load_events(target_date=target_date, outlet=outlet,
                       window_days=window_days)


def summarize_events(events: List[Dict]) -> str:
    """Human-readable summary of events for use in LLM prompts."""
    if not events:
        return "Нет заранее известных событий."
    lines = []
    for ev in events:
        date_str = ev["date"].strftime("%d.%m.%Y") if ev.get("date") else "?"
        lines.append(f"- {date_str}: [{ev['event_type']}] {ev['description']}")
    return "\n".join(lines)
