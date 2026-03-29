"""
Event calendar loader: reads events.csv and filters by date + outlet.
"""
import datetime
import os
from typing import List, Dict, Optional

import pandas as pd

from config import EVENTS_CSV, OUTLETS


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

    # Filter by date window
    if target_date is not None:
        lo = target_date - datetime.timedelta(days=window_days)
        hi = target_date + datetime.timedelta(days=window_days)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)]

    # Filter by outlet
    if outlet is not None:
        df = df[df["outlets"].str.contains(outlet, na=False)]

    events = []
    for _, row in df.iterrows():
        events.append({
            "date":        row["date"],
            "event_type":  row.get("event_type", ""),
            "description": row.get("description", ""),
            "outlets":     [o.strip() for o in str(row.get("outlets", "")).split(";")],
            "source":      row.get("source", ""),
        })

    return events


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
