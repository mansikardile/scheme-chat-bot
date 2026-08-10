from pathlib import Path
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

X_API_KEY = "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc"
BASE_URL = "https://api.myscheme.gov.in"

HEADERS = {
    "x-api-key": X_API_KEY,
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.myscheme.gov.in",
    "Referer": "https://www.myscheme.gov.in/",
    "User-Agent": "Mozilla/5.0",
}

session = requests.Session()
session.headers.update(HEADERS)

retry_strategy = Retry(
    total=3, 
    backoff_factor=1, 
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_total_schemes():
    r = session.get(f"{BASE_URL}/search/v6/schemes/facets", params={"lang": "en"})
    r.raise_for_status()
    return r.json()["data"]["summary"]["total"]

def search_schemes(start_index, limit=100):
    params = {
        "lang": "en",
        "q": "[]",
        "keyword": "",
        "sort": "schemename-asc",
        "from": start_index,
        "size": limit,
    }

    r = session.get(f"{BASE_URL}/search/v6/schemes", params=params)
    r.raise_for_status()
    return r.json()["data"]["hits"]["items"]

def fetch_scheme_details(slug):
    params = {
        "lang": "en",
        "slug": slug,
    }
    r = session.get(f"{BASE_URL}/schemes/v6/public/schemes", params=params)
    r.raise_for_status()
    return r.json()["data"]

def save_all_schemes():
    total = get_total_schemes()

    saved = 0
    skipped = 0
    start_index = 0

    schemes_dir = Path(__file__).resolve().parents[2] / "schemes"
    schemes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Total Schemes available: {total}")

    while start_index < total:
        try:
            schemes = search_schemes(start_index)
        except requests.RequestException as e:
            print(f"Failed to fetch scheme list at index {start_index}: {e}")
            break

        if not schemes:
            print("No more schemes returned.")
            break

        for scheme in schemes:
            slug = scheme["fields"]["slug"]
            filepath = schemes_dir / f"{slug}.json"

            if filepath.exists():
                skipped += 1
                continue

            try:
                details = fetch_scheme_details(slug)
                if not details:
                    print(f"Skipping {slug}: null reponse")
                    skipped += 1
                    continue
                filepath.write_text(
                    json.dumps(details, ensure_ascii=False, indent=2))
                saved += 1
                print(f"Saved {slug}")
                
                time.sleep(0.25) 
            except requests.RequestException as e:
                print(f"Failed to download {slug} after retries: {e}")
                
        start_index += len(schemes)


if __name__ == "__main__":
    save_all_schemes()
