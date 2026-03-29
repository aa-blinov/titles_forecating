import datetime
from scraper import scrape_rss, OUTLETS

def test_rss_content():
    print("--- RSS CONTENT VERIFICATION ---")
    slugs = list(OUTLETS.keys())
    
    for slug in slugs:
        print(f"\nOUTLET: {OUTLETS[slug]['name']} ({slug})")
        try:
            recs = scrape_rss(slug)
            if not recs:
                print("  [!] No records found")
                continue
            
            # Print stats and sample for first 3 records
            print(f"  Total records: {len(recs)}")
            for i, rec in enumerate(recs[:3]):
                title = rec.get('title', 'N/A')
                lead = rec.get('lead')
                
                print(f"  {i+1}. Title: {title}")
                if lead:
                    print(f"     Lead Length: {len(lead)} characters")
                    print(f"     Lead Snippet: {lead[:150]}...")
                else:
                    print(f"     Lead: [EMPTY / None]")
                print("-" * 20)
                
        except Exception as e:
            print(f"  [ERROR] {slug}: {e}")

if __name__ == "__main__":
    test_rss_content()
