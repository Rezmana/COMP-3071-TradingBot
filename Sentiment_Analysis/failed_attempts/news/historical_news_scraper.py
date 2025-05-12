# multi_news_scraper.py

from bs4 import BeautifulSoup
from typing import List, Dict   
import argparse, datetime as dt
from dateutil.rrule import rrule, DAILY 
import cloudscraper
from urllib.parse import urlparse
import feedparser
import re, xml.etree.ElementTree as ET


# ----------------------------
# Define Each Site's Extractor
# ----------------------------

def extract_coindesk_range(start: dt.date, end: dt.date) -> List[Dict[str, str]]:
    """
    Parse CoinDesk's news-sitemap index, pick sitemap URLs whose YYYY-MM-DD
    falls in [start, end], then collect the <news:> items inside.
    About 300 URLs per year; ~5 MB total per year.
    """
    idx_xml = fetch_html(
        "https://www.coindesk.com/sitemap_news.xml"
    )
    idx_root = ET.fromstring(idx_xml)

    # regex pulls 2025-04-25-06 from …/sitemap-news-2025-04-25-06.xml
    date_re = re.compile(r"news-sitemap-(\d{4})-(\d{2})-(\d{2})")

    arts: List[Dict[str, str]] = []
    for sm in idx_root.findall(".//{*}loc"):
        m = date_re.search(sm.text)
        if not m:
            continue
        y, mth, d = map(int, m.groups())
        day = dt.date(y, mth, d)
        if not (start <= day <= end):
            continue

        xml = fetch_html(sm.text)
        root = ET.fromstring(xml)

        for url in root.findall(".//{*}url"):
            pubd = url.find(".//{*}publication_date")
            title = url.find(".//{*}title")
            loc   = url.find(".//{*}loc")
            if not (pubd is not None and title is not None and loc is not None):
                continue
            day_item = dt.date.fromisoformat(pubd.text[:10])
            if start <= day_item <= end:
                arts.append(
                    {
                        "title": title.text,
                        "snippet": "",
                        "url": loc.text,
                        "source": "CoinDesk",
                    }
                )
    print(f"[DEBUG] CoinDesk candidate sitemaps in window: {len(arts)}")
    return arts


def extract_cointelegraph_day(day: dt.date) -> List[Dict[str, str]]:
    url  = f"https://cointelegraph.com/archive/{day.isoformat()}"
    soup = BeautifulSoup(fetch_html(url), "html.parser")

    arts: List[Dict[str, str]] = []
    cards = soup.select("a.post-card-inline__figure-link") or \
            soup.select("a.post-card-inline__title-link")
    for a in cards:
        href = a["href"]
        if not href.startswith("http"):
            href = "https://cointelegraph.com" + href
        title = a.get_text(strip=True)
        if title:
            arts.append({"title": title, "snippet": "", "url": href,
                         "source": "CoinTelegraph"})
    return arts


def decrypt_sitemaps_between(start, end) -> List[str]:
    idx_xml = (fetch_html("https://decrypt.co/Sitemap_index.xml")
            or fetch_html("https://decrypt.co/sitemap_index.xml")
            or fetch_html("https://decrypt.co/wp-sitemap.xml"))

    soup = BeautifulSoup(idx_xml, "xml")
    urls = []
    for sm in soup.find_all("sitemap"):
        loc = sm.loc.string if sm.loc else ""
        if "post-sitemap" not in loc:
            continue
        #     post-sitemap-2021-04.xml → 2021-04
        m = re.search(r"(\d{4})-(\d{2})", loc)
        if not m:
            continue
        year, month = map(int, m.groups())
        first_day = dt.date(year, month, 1)
        last_day  = (first_day + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
        if first_day <= end and last_day >= start:
            urls.append(loc)
    print(f"[DEBUG] Decrypt monthly sitemaps selected: {len(urls)}")
    return urls


def extract_decrypt_range(start: dt.date, end: dt.date) -> List[Dict[str, str]]:
    arts: List[Dict[str, str]] = []
    for sm_url in decrypt_sitemaps_between(start, end):
        xml  = fetch_html(sm_url)
        soup = BeautifulSoup(xml, "xml")

        for item in soup.find_all("url"):
            date_tag  = item.find("news:publication_date")
            title_tag = item.find("news:title")
            loc_tag   = item.find("loc")
            if not (date_tag and title_tag and loc_tag):
                continue

            pubdate = dt.date.fromisoformat(date_tag.text[:10])
            if start <= pubdate <= end:
                arts.append(
                    {
                        "title":  title_tag.text,
                        "snippet": "",
                        "url":    loc_tag.text,
                        "source": "Decrypt",
                    }
                )
    return arts


# ----------------------------
# News Source Registry
# ----------------------------

NEWS_SOURCES = [
    {"name": "CoinDesk",      "extractor": extract_coindesk_range,   "mode": "range"},
    {"name": "CoinTelegraph", "extractor": extract_cointelegraph_day,"mode": "day"},
    {"name": "Decrypt",       "extractor": extract_decrypt_range,    "mode": "range"},
]


# ----------------------------
# Fetch Utility
# ----------------------------
scraper = cloudscraper.create_scraper(browser={"custom": "Mozilla/5.0"})

def fetch_html(url: str) -> str:
    try:
        r = scraper.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


# ----------------------------
# Main Scraping Pipeline
# ----------------------------

def get_all_news_articles(start: dt.date, end: dt.date) -> List[Dict[str, str]]:
    all_arts = []
    for src in NEWS_SOURCES:
        print(f"Fetching from {src['name']} …")
        if src["mode"] == "day":
            for d in rrule(DAILY, dtstart=start, until=end):
                all_arts.extend(src["extractor"](d.date()))
        else:  # Decrypt range
            all_arts.extend(src["extractor"](start, end))
        print(f"✓ {src['name']} total so far: {len(all_arts)}")
    return all_arts


# ----------------------------
# Example Use
# ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    st = dt.date.fromisoformat(args.start)
    ed = dt.date.fromisoformat(args.end)
    arts = get_all_news_articles(st, ed)
    
    print(f"\nFetched {len(arts)} articles in total.")
