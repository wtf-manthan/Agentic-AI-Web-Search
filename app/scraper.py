import re
import primp
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_urls(text: str) -> list[str]:
    return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)


def _collect_urls(search_results: dict) -> list[str]:
    """Pull every URL out of whatever shape the search results are in."""
    urls = []
    for _, results in search_results.items():
        if not results:
            continue
        if isinstance(results, dict) and "results" in results:
            for item in results["results"]:
                if isinstance(item, dict) and "url" in item:
                    urls.append(item["url"])
        elif isinstance(results, str):
            urls.extend(extract_urls(results))
    return urls


# ---------------------------------------------------------------------------
# Single-URL scraper — used by search_and_index_node
# ---------------------------------------------------------------------------

def scrape_and_clean_url(url: str) -> str | None:
    """
    Scrapes a single URL with primp (TLS browser impersonation) + BeautifulSoup,
    or returns None if inaccessible / blocked / yields too little content.
    """
    try:
        client = primp.Client(
            impersonate="random",
            follow_redirects=True,
            timeout=4,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        res = client.get(url)
        
        if res.status_code != 200:
            return None
            
        html = res.text
        
        # Check for common anti-bot error pages before parsing
        anti_bot_strings = [
            "Access Denied", 
            "Just a moment...", 
            "Enable JavaScript and cookies to continue", 
            "Attention Required! | Cloudflare",
            "errors.edgesuite.net"
        ]
        if any(bad in html[:2000] for bad in anti_bot_strings):
            print(f"    [Scraper] ❌ Bot protection detected on {url}")
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Strip noise elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg", "form", "button", "ad"]):
            tag.decompose()
            
        text = " ".join(soup.get_text().split())
        
        if len(text) > 50_000:
            text = text[:50_000] + "..."
            
        if len(text) > 100:
            print(f"    [Scraper] ✅ Scraped {url} ({len(text)} chars)")
            return text
        else:
            return None
            
    except Exception as e:
        print(f"    [Scraper] ❌ Failed {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch scraper — used if scraping a bundle of search results directly
# ---------------------------------------------------------------------------

def scrape_urls_from_results(search_results: dict, max_sites: int = 5) -> list[str]:
    urls = _collect_urls(search_results)

    if not urls:
        print("No URLs found to scrape.")
        return []

    urls = urls[:max_sites]
    scraped_content = []

    for url in urls:
        text = scrape_and_clean_url(url)
        if text:
            scraped_content.append(f"Content from {url}:\n{text}")

    return scraped_content
