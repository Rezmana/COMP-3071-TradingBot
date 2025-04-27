# wayback_scraper.py
import argparse, datetime as dt, json, re, time, requests, pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict
from requests.adapters import HTTPAdapter, Retry

CDX = "https://web.archive.org/cdx/search/cdx"

SITES = {
    "CoinDesk":      r"coindesk.com/*",
    "CoinTelegraph": r"cointelegraph.com/*",
    "Decrypt":       r"decrypt.co/*",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def list_snapshots(site_pattern: str,
                   start: str,
                   end: str,
                   per_page: int = 2000) -> List[Dict]:
    """
    Return [{ts,url},…] for snapshots in [start,end], looping one day at a time
    so we never hit a 504.  Each daily query paginates with page=0,1,2… .
    """
    s_date = dt.date.fromisoformat(start)
    e_date = dt.date.fromisoformat(end)
    all_rows: List[Dict] = []

    sess = requests.Session()
    sess.headers.update(HEADERS)
    retries = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1))
    sess.mount("https://", retries)

    day = s_date
    while day <= e_date:
        day_str = day.isoformat()
        params = {
            "url": site_pattern,
            "from": day_str,
            "to":   day_str,
            "output": "json",
            "filter": "statuscode:200",
            "collapse": "urlkey",
            "fl": "timestamp,original",
            "pageSize": str(per_page),
            "gzip": "false",
        }
        page = 0
        while True:
            params["page"] = str(page)
            try:
                r = sess.get(CDX, params=params, timeout=40)
                r.raise_for_status()
                rows = r.json()
                if rows and rows[0][0] == "timestamp":
                    rows = rows[1:]
                if not rows:
                    break
                all_rows.extend({"ts": x[0], "url": x[1]} for x in rows)
                page += 1
            except requests.HTTPError as e:
                # 504 or 502 on this page? wait & retry next page
                if r.status_code in (502, 504):
                    time.sleep(3)
                    continue
                raise
        day += dt.timedelta(days=1)
    return all_rows

def fetch_html_via_wayback(ts: str, url: str) -> str:
    wb_url = f"https://web.archive.org/web/{ts}/{url}"
    resp = requests.get(wb_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text

def parse_title(site: str, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if site == "CoinDesk":
        h = soup.find("h1") or soup.find("title")
    elif site == "CoinTelegraph":
        h = soup.find("h1") or soup.find("title")
    else:  # Decrypt
        h = soup.find("h1") or soup.find("title")
    return h.get_text(strip=True) if h else ""

def scrape_site(site: str, pattern: str, start: str, end: str) -> List[Dict]:
    print(f"CDX query for {site} …")
    snaps = list_snapshots(pattern, start, end)
    print(f"  {len(snaps)} unique URLs in window")
    arts = []
    for snap in snaps:
        try:
            html = fetch_html_via_wayback(snap["ts"], snap["url"])
            title = parse_title(site, html)
            if title:
                arts.append(
                    {
                        "date": snap["ts"][:8],  # YYYYMMDD
                        "title": title,
                        "url": snap["url"],
                        "source": site,
                    }
                )
        except Exception:
            pass
        time.sleep(0.1)  # be polite to archive.org
    return arts

def main(start: str, end: str):
    all_rows: List[Dict] = []
    for site, patt in SITES.items():
        all_rows.extend(scrape_site(site, patt, start, end))
    df = pd.DataFrame(all_rows)
    print(f"\nFetched {len(df)} articles total.")
    df.to_csv(f"wayback_articles_{start}_{end}.csv", index=False)
    print("Saved to CSV; sample:")
    print(df.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    main(args.start, args.end)
