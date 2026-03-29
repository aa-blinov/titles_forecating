import datetime
from scraper import scrape_rss, scrape_outlet, OUTLETS

def test_scrapers():
    slugs = list(OUTLETS.keys())
    print(f"Available outlets: {slugs}")
    
    results = {}
    
    # Test 1: Quick RSS test for each outlet
    print("\n--- TEST 1: RSS FEEDS ---")
    for slug in slugs:
        try:
            print(f"Testing RSS for {slug}...")
            recs = scrape_rss(slug)
            print(f"  [rss] Found {len(recs)} recent entries")
            if recs:
                print(f"  [rss] Example record keys: {list(recs[0].keys())}")
                print(f"  [rss] Example title: {recs[0]['title']}")
            results[slug] = recs
        except Exception as e:
            print(f"  [rss] FAILED for {slug}: {e}")

    # Test 2: Full scrape for one outlet (limited range)
    if slugs:
        test_slug = slugs[0]
        print(f"\n--- TEST 2: FULL SCRAPE (1 DAY) for {test_slug} ---")
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        try:
            full_recs = scrape_outlet(test_slug, start_date=yesterday, end_date=yesterday)
            print(f"  [full] Total records for {test_slug} on {yesterday}: {len(full_recs)}")
        except Exception as e:
            print(f"  [full] FAILED for {test_slug}: {e}")

if __name__ == "__main__":
    test_scrapers()
