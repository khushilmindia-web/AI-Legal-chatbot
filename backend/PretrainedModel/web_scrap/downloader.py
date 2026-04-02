import json
import requests
import os
from urllib.parse import urlparse, urlunparse, urljoin, parse_qs
import time
from tqdm import tqdm
import random
from bs4 import BeautifulSoup
import re
import hashlib
from backend.paths import DATA_DIR

with open(DATA_DIR / "Json" / "acts_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_pdf_urls = set()
for item in data:
    for pdf_url in item["pdfs"]:
        all_pdf_urls.add(pdf_url)

all_pdf_urls = list(all_pdf_urls)
print(f"Total unique PDFs found: {len(all_pdf_urls)}")

os.makedirs(DATA_DIR / "pdfs", exist_ok=True)
os.makedirs(DATA_DIR / "logs", exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

session = requests.Session()
session.headers.update(headers)

def clean_url(url):
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned)

def extract_pdf_from_indiacode(html_content, base_url):
    soup = BeautifulSoup(html_content, 'lxml')
    
    pdf_link = None
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/bitstream/' in href and href.lower().endswith('.pdf'):
            pdf_link = urljoin(base_url, href)
            #break
        if href.lower().endswith('.pdf'):
            pdf_link = urljoin(base_url, href)
            #break
    
    if not pdf_link:
        meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})
        if meta and meta.get('content'):
            pdf_link = meta['content']
            return pdf_link

def download_pdf(url, save_path, max_retries=3):
    url = clean_url(url)
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=60, allow_redirects=True, stream=True)
            final_url = response.url
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'application/pdf' in content_type:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True, f"Direct PDF from {final_url}"
            
            elif 'text/html' in content_type:
                if 'indiacode.nic.in/handle/' in final_url or 'indiacode.nic.in/bitstream/' in final_url:
                    pdf_link = extract_pdf_from_indiacode(response.text, final_url)
                    if pdf_link:
                        pdf_response = session.get(pdf_link, timeout=60, stream=True)
                        if 'application/pdf' in pdf_response.headers.get('Content-Type', '').lower():
                            with open(save_path, 'wb') as f:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            return True, f"PDF extracted from indiacode: {pdf_link}"
                        else:
                            return False, f"Extracted link not PDF: {pdf_link}"
                    else:
                        soup = BeautifulSoup(response.text, 'lxml')
                        for a in soup.find_all('a', href=True):
                            if a['href'].lower().endswith('.pdf'):
                                pdf_link = urljoin(final_url, a['href'])
                                pdf_response = session.get(pdf_link, timeout=60, stream=True)
                                if 'application/pdf' in pdf_response.headers.get('Content-Type', '').lower():
                                    with open(save_path, 'wb') as f:
                                        for chunk in pdf_response.iter_content(chunk_size=8192):
                                            if chunk:
                                                f.write(chunk)
                                    return True, f"PDF found in generic link: {pdf_link}"
                        return False, "No PDF link found in HTML"
                else:
                    soup = BeautifulSoup(response.text, 'lxml')
                    for a in soup.find_all('a', href=True):
                        if a['href'].lower().endswith('.pdf'):
                            pdf_link = urljoin(final_url, a['href'])
                            pdf_response = session.get(pdf_link, timeout=60, stream=True)
                            if 'application/pdf' in pdf_response.headers.get('Content-Type', '').lower():
                                with open(save_path, 'wb') as f:
                                    for chunk in pdf_response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                return True, f"PDF found in HTML: {pdf_link}"
                    return False, "HTML page contains no PDF link"
            
            elif response.status_code == 404:
                return False, "404 Not Found"
            else:
                return False, f"Unexpected content type: {content_type} (status {response.status_code})"
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = random.uniform(3, 7)
                print(f"  Error: {e}, retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                return False, f"Request failed: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    return False, "Max retries exceeded"

failed_log = open(DATA_DIR / "logs" / "failed_pdfs.txt", "a", encoding="utf-8")
success_count = 0
fail_count = 0
skipped_legislative = 0
def url_exists(url):
    try:
        r = session.head(url, timeout=10, allow_redirects=True)
        return r.status_code == 200
    except:
        return False

for pdf_url in tqdm(all_pdf_urls, desc="Downloading PDFs"):
    hash_name = hashlib.md5(pdf_url.encode()).hexdigest()
    filename = hash_name + ".pdf"
    save_path = os.path.join(DATA_DIR / "pdfs", filename)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:  
        continue
    success, message = download_pdf(pdf_url, save_path)
    if success:
        success_count += 1
    else:
        fail_count += 1
        failed_log.write(f"{pdf_url}\t{message}\n")
        if os.path.exists(save_path) and os.path.getsize(save_path) == 0:
            os.remove(save_path)
    
    time.sleep(random.uniform(2, 5))

failed_log.close()
print(f"\n Successfully downloaded: {success_count}")
print(f" Failed: {fail_count}")
if skipped_legislative > 0:
    print(f"Skipped legislative.gov.in: {skipped_legislative}")
print("Failed URLs logged in logs/failed_pdfs.txt")
