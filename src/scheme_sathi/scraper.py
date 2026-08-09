from pathlib import Path
import json
import requests

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
        schemes = search_schemes(start_index)

        if not schemes:
            print("No more schemes returned.")
            break

        for scheme in schemes:
            slug = scheme["fields"]["slug"]
            details = fetch_scheme_details(slug)
            filepath = schemes_dir / f"{slug}.json"

            if filepath.exists():
                skipped += 1
                continue
            
            filepath.write_text(json.dumps(details, ensure_ascii=False, indent=2))
            saved += 1
            print(f"Saved {slug}")
            
        start_index += len(schemes)


if __name__ == "__main__":
    save_all_schemes()
