"""
HAGA Folder RAG — Scraper
===========================
Downloadt alle patiëntenfolders van folders.hagaziekenhuis.nl.

Twee strategieën:
  1. Parse de index-pagina (hagaziekenhuis.nl/patientenfolders)
  2. Fallback: brute-force ID scan (1000-6000)

Gebruik:
    python scrape_folders.py --output ./folders/
    python scrape_folders.py --output ./folders/ --method index
    python scrape_folders.py --output ./folders/ --method bruteforce --start 1000 --end 6000
"""

import argparse
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = "https://folders.hagaziekenhuis.nl"
INDEX_URL = "https://www.hagaziekenhuis.nl/patientenfolders"
PDF_PATH_PATTERN = "/patientenfolders/{id}-{slug}.pdf"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Rate limiting: wees respectvol naar de HAGA-server
DELAY_BETWEEN_REQUESTS = 0.5  # seconden
MAX_WORKERS = 4               # parallelle downloads


def get_session() -> requests.Session:
    """Maak een session met retry-logica."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


# ─────────────────────────────────────────────
# STRATEGIE 1: Index-pagina parsen
# ─────────────────────────────────────────────

def scrape_index(session: requests.Session) -> list[str]:
    """
    Probeer de index-pagina te parsen voor PDF-links.
    Returns lijst van volledige PDF URLs.
    """
    print("📋 Strategie 1: Index-pagina parsen...")

    try:
        # Probeer eerst de hoofdsite
        resp = session.get(INDEX_URL, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Zoek alle PDF-links
        # Patronen die we verwachten:
        # - /patientenfolders/1234-slug.pdf
        # - folders.hagaziekenhuis.nl/patientenfolders/1234-slug.pdf
        pdf_pattern = re.compile(
            r'(?:folders\.hagaziekenhuis\.nl)?/patientenfolders/(\d+-[^"\'>\s]+\.pdf)',
            re.IGNORECASE,
        )
        matches = pdf_pattern.findall(html)
        urls = list(set(f"{BASE_URL}/patientenfolders/{m}" for m in matches))

        if urls:
            print(f"  ✓ {len(urls)} PDF-links gevonden op index-pagina")
            return urls
        else:
            print("  ⚠ Geen PDF-links gevonden op index (JS-rendered pagina?)")
            return []

    except Exception as e:
        print(f"  ⚠ Index-pagina niet bereikbaar: {e}")
        return []


# ─────────────────────────────────────────────
# STRATEGIE 2: Sitemap crawlen
# ─────────────────────────────────────────────

def scrape_sitemap(session: requests.Session) -> list[str]:
    """Probeer PDF-links uit de sitemap te halen."""
    print("📋 Strategie 2: Sitemap crawlen...")

    sitemap_urls = [
        f"{BASE_URL}/sitemap.xml",
        "https://www.hagaziekenhuis.nl/sitemap.xml",
        f"{BASE_URL}/robots.txt",
    ]

    pdf_urls = []
    for sitemap_url in sitemap_urls:
        try:
            resp = session.get(sitemap_url, timeout=15)
            if resp.status_code == 200:
                content = resp.text
                # Zoek PDF URLs
                pdf_pattern = re.compile(
                    r'(https?://folders\.hagaziekenhuis\.nl/patientenfolders/\d+-[^<"\s]+\.pdf)',
                    re.IGNORECASE,
                )
                matches = pdf_pattern.findall(content)
                pdf_urls.extend(matches)
                print(f"  ✓ {len(matches)} links in {sitemap_url}")
        except Exception as e:
            print(f"  ⚠ {sitemap_url}: {e}")

    pdf_urls = list(set(pdf_urls))
    if pdf_urls:
        print(f"  ✓ Totaal {len(pdf_urls)} unieke PDF-links uit sitemaps")
    return pdf_urls


# ─────────────────────────────────────────────
# STRATEGIE 3: Brute-force ID scan
# ─────────────────────────────────────────────

def bruteforce_scan(
    session: requests.Session,
    start_id: int = 1000,
    end_id: int = 6500,
) -> list[str]:
    """
    Probeer elke ID en kijk welke een geldige PDF teruggeven.
    Gebruikt HEAD requests voor snelheid.
    """
    print(f"📋 Strategie 3: Brute-force scan IDs {start_id}-{end_id}...")

    valid_urls = []
    checked = 0
    total = end_id - start_id

    def check_id(folder_id: int) -> str | None:
        """Check of een ID een geldige PDF-pagina heeft."""
        # Eerst proberen we de redirect-URL
        url = f"{BASE_URL}/{folder_id}"
        try:
            resp = session.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                final_url = resp.url
                if ".pdf" in final_url.lower():
                    return final_url
                # Soms redirected het naar een HTML-pagina met PDF-link
                # Probeer dan de directe PDF URL
                resp2 = session.get(url, timeout=10)
                if resp2.status_code == 200:
                    # Zoek PDF link in response
                    pdf_match = re.search(
                        r'/patientenfolders/(\d+-[^"\'>\s]+\.pdf)',
                        resp2.text,
                    )
                    if pdf_match:
                        return f"{BASE_URL}/patientenfolders/{pdf_match.group(1)}"

            # Probeer ook het ?alttemplate=PDFPatientenfolder patroon
            pdf_url = f"{BASE_URL}/{folder_id}?alttemplate=PDFPatientenfolder&toegankelijk=1"
            resp = session.head(pdf_url, timeout=10)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" in content_type:
                    return pdf_url

        except Exception:
            pass
        return None

    # Sequentieel maar snel (HEAD requests)
    for folder_id in range(start_id, end_id + 1):
        result = check_id(folder_id)
        if result:
            valid_urls.append(result)
            print(f"  ✓ ID {folder_id}: gevonden ({len(valid_urls)} totaal)")

        checked += 1
        if checked % 500 == 0:
            print(f"  ... {checked}/{total} IDs gecontroleerd, {len(valid_urls)} gevonden")

        time.sleep(0.1)  # Respectvol naar de server

    print(f"  ✓ {len(valid_urls)} geldige folders gevonden")
    return valid_urls


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

def download_pdf(session: requests.Session, url: str, output_dir: Path) -> bool:
    """Download een enkele PDF."""
    try:
        # Bepaal filename
        # URL: .../patientenfolders/1234-slug.pdf of .../1234?alttemplate=...
        if ".pdf" in url:
            filename = url.split("/")[-1]
        else:
            # Extract ID en maak een filename
            match = re.search(r'/(\d+)', url)
            if match:
                filename = f"{match.group(1)}.pdf"
            else:
                filename = url.replace("/", "_").replace(":", "")[-50:] + ".pdf"

        filepath = output_dir / filename

        # Skip als al gedownload
        if filepath.exists() and filepath.stat().st_size > 1000:
            return True

        resp = session.get(url, timeout=30)
        resp.raise_for_status()

        # Controleer of het echt een PDF is
        content_type = resp.headers.get("Content-Type", "")
        is_pdf = (
            "pdf" in content_type
            or resp.content[:5] == b"%PDF-"
        )

        if is_pdf:
            filepath.write_bytes(resp.content)
            return True
        else:
            return False

    except Exception as e:
        return False


def download_all(
    session: requests.Session,
    urls: list[str],
    output_dir: Path,
    max_workers: int = MAX_WORKERS,
):
    """Download alle PDFs met progress tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n⬇ {len(urls)} PDFs downloaden naar {output_dir}...")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for url in urls:
            future = executor.submit(download_pdf, session, url, output_dir)
            futures[future] = url
            time.sleep(DELAY_BETWEEN_REQUESTS)

        for future in as_completed(futures):
            url = futures[future]
            if future.result():
                success += 1
            else:
                failed += 1

            total = success + failed
            if total % 50 == 0:
                print(f"  📥 {total}/{len(urls)} — ✓ {success} ✗ {failed}")

    print(f"\n✅ Download klaar: {success} gelukt, {failed} mislukt")
    print(f"   Locatie: {output_dir.absolute()}")


# ─────────────────────────────────────────────
# ALTERNATIVE: Direct van hagaziekenhuis.nl/patientenfolders
# ─────────────────────────────────────────────

def scrape_via_api(session: requests.Session) -> list[str]:
    """
    Veel ziekenhuissites draaien op Umbraco CMS.
    Probeer bekende API-endpoints.
    """
    print("📋 Strategie 4: CMS API proberen...")

    api_endpoints = [
        f"{BASE_URL}/umbraco/api/patientenfolders/getall",
        f"{BASE_URL}/api/patientenfolders",
        "https://www.hagaziekenhuis.nl/umbraco/api/search?query=pdf&type=patientenfolder",
    ]

    for endpoint in api_endpoints:
        try:
            resp = session.get(endpoint, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✓ API response van {endpoint}")
                # Extract PDF URLs uit de response
                urls = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, str) and ".pdf" in value:
                                    urls.append(value)
                return urls
        except Exception:
            pass

    print("  ⚠ Geen werkende API gevonden")
    return []


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HAGA Folder Scraper")
    parser.add_argument("--output", default="./folders/", help="Output directory")
    parser.add_argument(
        "--method",
        choices=["auto", "index", "sitemap", "bruteforce", "api"],
        default="auto",
        help="Scraping strategie",
    )
    parser.add_argument("--start", type=int, default=1000, help="Start ID voor bruteforce")
    parser.add_argument("--end", type=int, default=6500, help="Eind ID voor bruteforce")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallelle downloads")
    args = parser.parse_args()

    output_dir = Path(args.output)
    session = get_session()

    pdf_urls = []

    if args.method == "auto":
        # Probeer alle strategieën
        pdf_urls = scrape_index(session)
        if not pdf_urls:
            pdf_urls = scrape_sitemap(session)
        if not pdf_urls:
            pdf_urls = scrape_via_api(session)
        if not pdf_urls:
            print("\n⚡ Automatische detectie leverde niets op.")
            print("   Stap over naar brute-force scan (dit duurt ~30 min)...")
            pdf_urls = bruteforce_scan(session, args.start, args.end)

    elif args.method == "index":
        pdf_urls = scrape_index(session)
    elif args.method == "sitemap":
        pdf_urls = scrape_sitemap(session)
    elif args.method == "api":
        pdf_urls = scrape_via_api(session)
    elif args.method == "bruteforce":
        pdf_urls = bruteforce_scan(session, args.start, args.end)

    if not pdf_urls:
        print("❌ Geen folders gevonden. Probeer --method bruteforce")
        return

    # Deduplicate
    pdf_urls = list(set(pdf_urls))
    print(f"\n📊 {len(pdf_urls)} unieke PDF URLs gevonden")

    # Download
    download_all(session, pdf_urls, output_dir, max_workers=args.workers)

    # Bewaar URL-lijst voor referentie
    url_file = output_dir / "_urls.txt"
    url_file.write_text("\n".join(sorted(pdf_urls)))
    print(f"📝 URL-lijst opgeslagen: {url_file}")


if __name__ == "__main__":
    main()
