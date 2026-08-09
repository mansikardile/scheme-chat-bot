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
