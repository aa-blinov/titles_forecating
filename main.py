"""
Main CLI for the news forecast pipeline.

Usage:
  python main.py --mode scrape     # collect 90 days of news
  python main.py --mode etl        # clean + deduplicate
  python main.py --mode analyze    # topics + noise check + entities
  python main.py --mode backtest   # holdout backtest
  python main.py --mode metrics    # evaluate backtest quality
  python main.py --mode forecast   # generate forecast for 02.04.2026
  python main.py --mode all        # run full pipeline end-to-end

Optional flags:
  --outlets rbc,kommersant,...     # comma-separated outlet slugs (default: all 5)
  --target  2026-04-02             # override target forecast date
  --no-llm                         # skip LLM generation step
  --enrich-leads                   # fetch full article leads during scraping (slow)
"""
import argparse
import datetime
import sys
import os

# Ensure titles/ is on sys.path regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="News Forecast Pipeline — 5 Russian media outlets"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["scrape", "etl", "analyze", "backtest", "metrics", "forecast", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--outlets", default=None,
        help="Comma-separated outlet slugs: rbc,kommersant,vedomosti,lenta,interfax",
    )
    parser.add_argument(
        "--target", default="2026-04-02",
        help="Forecast target date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip Ollama LLM generation",
    )
    parser.add_argument(
        "--enrich-leads", action="store_true",
        help="Fetch full leads for each article during scraping (slow)",
    )
    return parser.parse_args()


def _resolve_outlets(outlets_arg: str) -> list:
    from config import OUTLET_SLUGS
    if not outlets_arg:
        return OUTLET_SLUGS
    requested = [s.strip().lower() for s in outlets_arg.split(",")]
    invalid   = [s for s in requested if s not in OUTLET_SLUGS]
    if invalid:
        print(f"[main] Unknown outlets: {invalid}. Valid: {OUTLET_SLUGS}")
        sys.exit(1)
    return requested


# ================================================================
#  STAGE RUNNERS
# ================================================================

def run_scrape(outlets: list, enrich_leads: bool) -> None:
    print("\n========== STAGE: SCRAPE ==========")
    from scraper import scrape_all
    from config import HISTORY_FROM, TODAY
    results = scrape_all(
        slugs=outlets,
        start_date=HISTORY_FROM,
        end_date=TODAY,
        enrich_leads=enrich_leads,
    )
    for slug, recs in results.items():
        print(f"  {slug}: {len(recs)} records saved to data/raw/")


def run_etl(outlets: list) -> None:
    print("\n========== STAGE: ETL ==========")
    from etl import run_etl as _run_etl
    _run_etl(slugs=outlets)


def run_analyze(outlets: list) -> None:
    print("\n========== STAGE: ANALYZE ==========")
    from analyzer import analyze_outlet
    for slug in outlets:
        analyze_outlet(slug)


def run_backtest(outlets: list) -> None:
    print("\n========== STAGE: BACKTEST ==========")
    from backtester import backtest_all
    backtest_all(slugs=outlets)


def run_metrics(outlets: list) -> None:
    print("\n========== STAGE: METRICS ==========")
    from metrics import evaluate_all
    reports = evaluate_all(slugs=outlets)
    print("\n--- Summary ---")
    for slug, rep in reports.items():
        print(
            f"  {slug:<12} "
            f"topic_hit={rep.get('topic_hit_rate', 0):.3f}  "
            f"entity_f1={rep.get('entity_match_f1', 0):.3f}  "
            f"sem_sim={rep.get('semantic_similarity', 0):.3f}  "
            f"style={rep.get('style_match', 0):.3f}  "
            f"diversity={rep.get('diversity_score', 0):.3f}"
        )


def run_forecast(outlets: list, target_date: datetime.date,
                 use_llm: bool) -> None:
    print(f"\n========== STAGE: FORECAST → {target_date} ==========")
    from forecaster import forecast_all
    from config import FORECASTS_DIR
    reports = forecast_all(slugs=outlets, target_date=target_date, use_llm=use_llm)

    # Pretty print summary
    date_str  = str(target_date)
    json_path = os.path.join(FORECASTS_DIR, f"forecast_{date_str}.json")

    for slug, rep in reports.items():
        name = rep.get("outlet_name", slug)
        preds = rep.get("predictions", [])
        llm   = [p for p in preds if p.get("method") == "llm"]
        cal   = [p for p in preds if p.get("method") == "calendar"]
        base  = [p for p in preds if p.get("method") in ("inertia", "frequency")]

        print(f"\n  [{name}]")
        print(f"    Top topics : {', '.join(rep.get('top_topics', [])[:3])}")
        print(f"    Calendar   : {len(cal)} events")
        print(f"    Baselines  : {len(base)} predictions")
        print(f"    LLM titles : {len(llm)}")

        # Print LLM headlines if any
        for item in llm[:5]:
            title = item.get("title", "")
            lead  = item.get("lead", "")
            if title:
                print(f"      • {title}")
                if lead:
                    print(f"        > {lead[:120]}")

        # Print calendar items
        for item in cal[:3]:
            title = item.get("title", "")
            if title:
                print(f"      [cal] {title}")

    print(f"\n  Full report saved → {json_path}")


# ================================================================
#  MAIN
# ================================================================

def main():
    args     = _parse_args()
    outlets  = _resolve_outlets(args.outlets)
    use_llm  = not args.no_llm

    try:
        target_date = datetime.date.fromisoformat(args.target)
    except ValueError:
        print(f"[main] Invalid date: {args.target}. Use YYYY-MM-DD.")
        sys.exit(1)

    print(f"[main] Mode={args.mode}  Outlets={outlets}  Target={target_date}  LLM={use_llm}")

    if args.mode in ("scrape", "all"):
        run_scrape(outlets, args.enrich_leads)

    if args.mode in ("etl", "all"):
        run_etl(outlets)

    if args.mode in ("analyze", "all"):
        run_analyze(outlets)

    if args.mode in ("backtest", "all"):
        run_backtest(outlets)

    if args.mode in ("metrics", "all"):
        run_metrics(outlets)

    if args.mode in ("forecast", "all"):
        run_forecast(outlets, target_date, use_llm)

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
