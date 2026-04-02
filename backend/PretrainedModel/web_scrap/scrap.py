import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
from backend.paths import DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def is_pdf_link(href):
    if not href:
        return False
    href_lower = href.lower()
    return href_lower.endswith('.pdf') or '/pdf/' in href_lower or 'bitstream' in href_lower

def extract_page_data(url):
    try:
        print(f"Scraping: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} – skipping")
            return None
        
        soup = BeautifulSoup(resp.content, "lxml")
        
        title = None
        h1 = soup.find("h1")
        if h1 and h1.text.strip():
            title = h1.text.strip()
        else:
            h2 = soup.find("h2")
            if h2 and h2.text.strip():
                title = h2.text.strip()
            else:
                title_tag = soup.find("title")
                if title_tag and title_tag.text.strip():
                    title = title_tag.text.strip()
                else:
                    title = "No Title"
        
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a['href']
            if is_pdf_link(href):

                absolute_url = urljoin(url, href)
                pdf_links.append(absolute_url)
        
        pdf_links = list(dict.fromkeys(pdf_links))
        
        return {
            "url": url,
            "title": title,
            "pdfs": pdf_links
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    urls = [
        "https://nalsa.gov.in/faqs/",
        "https://nalsa.gov.in/the-legal-services-authorities-act-1987/",
        "https://nalsa.gov.in/rules/",
        "https://nalsa.gov.in/regulations/",
        "https://nalsa.gov.in/preventive-strategic-legal-services-schemes/",
        "https://nalsa.gov.in/the-commercial-courts-acts-rules/",
        "https://nalsa.gov.in/women-and-law/",
        "https://nalsa.gov.in/important-bare-acts/",
        "https://nalsa.gov.in/guidelines/",
        "https://nalsa.gov.in/legal-aid/",
        "https://www.indiacode.nic.in/handle/123456789/2455/",
        "https://www.indiacode.nic.in/handle/123456789/2493/",
        "https://www.indiacode.nic.in/repealed-act/repealed-act.jsp",
        ]

    data = []
    for url in urls:
        result = extract_page_data(url)
        if result:
            data.append(result)
        time.sleep(2) 

    os.makedirs(DATA_DIR / "Json", exist_ok=True)
    with open(DATA_DIR / "Json" / "acts_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(data)} entries to {DATA_DIR / 'Json' / 'acts_data.json'}")
    
    for item in data:
        print(f"{item['title'][:50]}... – {len(item['pdfs'])} PDFs")

if __name__ == "__main__":
    main()
    
