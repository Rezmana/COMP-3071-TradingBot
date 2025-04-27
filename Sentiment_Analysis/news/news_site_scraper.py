import requests
from bs4 import BeautifulSoup
from typing import List, Dict   
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------------------
# Define Each Site's Extractor
# ----------------------------

def extract_coindesk_articles(_: str = None) -> List[Dict[str, str]]:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--ignore-certificate-errors")  # Optional for SSL errors
    driver = webdriver.Chrome(options=opts)

    articles = []
    try:
        driver.get("https://www.coindesk.com/")

        # Wait for the <h2> headlines to be visible
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a.text-color-charcoal-900"))
        )

        cards = driver.find_elements(By.CSS_SELECTOR, "a.text-color-charcoal-900")
        for card in cards:
            try:
                title_elem = card.find_element(By.TAG_NAME, "h2")
                title = title_elem.text.strip()
                url = card.get_attribute("href")
                if title and url:
                    articles.append({
                        "title": title,
                        "snippet": "",  # Add snippet scraping later if needed
                        "url": url,
                        "source": "CoinDesk"
                    })
            except Exception:
                continue

    finally:
        driver.quit()

    return articles


def extract_cointelegraph_articles() -> List[Dict[str, str]]:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)

    articles = []
    try:
        driver.get("https://cointelegraph.com/")

        # Accept cookie banner if needed
        try:
            consent = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            consent.click()
        except:
            pass  # No consent needed

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a.block.hover\\:underline"))
        )

        cards = driver.find_elements(By.CSS_SELECTOR, "a.block.hover\\:underline")
        for card in cards:
            try:
                title_elem = card.find_element(By.CSS_SELECTOR, "span[data-testid='post-card-title']")
                title = title_elem.text.strip()
                url = card.get_attribute("href")
                if title and url:
                    articles.append({
                        "title": title,
                        "snippet": "",  # Optional: can scrape more
                        "url": url,
                        "source": "CoinTelegraph"
                    })
            except Exception:
                continue

    finally:
        driver.quit()

    return articles



def extract_decrypt_articles() -> List[Dict[str, str]]:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)

    articles = []
    try:
        driver.get("https://decrypt.co/")

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h4.text-base"))
        )

        titles = driver.find_elements(By.CSS_SELECTOR, "h4.text-base")
        for title_h4 in titles:
            try:
                title = title_h4.text.strip()
                parent = title_h4.find_element(By.XPATH, "./ancestor::a[1]")
                url = parent.get_attribute("href")
                if title and url:
                    articles.append({
                        "title": title,
                        "snippet": "",
                        "url": url,
                        "source": "Decrypt"
                    })
            except Exception:
                continue

    finally:
        driver.quit()

    return articles


# ----------------------------
# News Source Registry
# ----------------------------

NEWS_SOURCES = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/",
        "extractor": lambda _: extract_coindesk_articles()
    },
    {
        "name": "CoinTelegraph",
        "url": "https://cointelegraph.com/",
        "extractor": extract_cointelegraph_articles
    },
    {
        "name": "Decrypt",
        "url": "https://decrypt.co/",
        "extractor": extract_decrypt_articles
    }
]


# ----------------------------
# Fetch Utility
# ----------------------------

def fetch_html(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


# ----------------------------
# Main Scraping Pipeline
# ----------------------------

def get_all_news_articles() -> List[Dict[str, str]]:
    all_articles = []

    for source in NEWS_SOURCES:
        print(f"Fetching from {source['name']}...")

        try:
            # If the extractor takes no arguments, assume Selenium is used
            if source["extractor"].__code__.co_argcount == 0:
                articles = source["extractor"]()
            else:
                html = fetch_html(source["url"])
                if not html:
                    print(f"⚠️ Failed to fetch HTML from {source['name']}")
                    continue
                articles = source["extractor"](html)

            print(f"✓ {len(articles)} articles from {source['name']}\n")
            all_articles.extend(articles)

        except Exception as e:
            print(f"❌ Error extracting from {source['name']}: {e}")

    return all_articles



# ----------------------------
# Example Use
# ----------------------------

if __name__ == "__main__":
    articles = get_all_news_articles()
    for article in articles[:5]:
        print(f"\n[{article['source']}] {article['title']}\n{article['snippet']}")