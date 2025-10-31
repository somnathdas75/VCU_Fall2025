"""
VCU Universal Scraper & Topic-based Fee Finder (single-file Streamlit app)

Features:
- Crawl VCU pages starting from a seed URL (breadth-first).
- Scan a local folder of saved HTML/PDF/TXT files.
- Extract tables via pandas.read_html and PDF text via pdfplumber/PyPDF2.
- Extract and group dollar amounts, find them near tuition-related anchors.
- Rank pages for a user topic and show grouped fees + context + program links.
- No hardcoded values; generalized topic search and automatic academics page fetch.

Run:
pip install streamlit requests beautifulsoup4 pandas lxml pdfplumber PyPDF2
streamlit run vcu_universal_scraper.py
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin, urlparse
from pathlib import Path

# Optional PDF libraries
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="VCU Universal Scraper & Topic Finder", layout="wide")

# ----------------- Constants & Regex -----------------
DOLLAR_RE = re.compile(r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
VCU_KEYWORDS = [
    "program", "programs", "undergraduate", "graduate", "majors",
    "degrees", "courses", "academics", "admissions", "tuition", "fees",
    "cost", "catalog", "curriculum", "study"
]
ANCHOR_KEYWORDS_GENERAL = [
    'non-virginia resident', 'non virginia resident', 'non-virginia', 'non-resident', 'nonresident',
    'out-of-state', 'out of state', 'tuition', 'tuition and fees', 'fees', 'cost', 'room', 'dining',
    'undergraduate', 'graduate', 'total', 'resident', 'in-state', 'out-of-state resident'
]

# ----------------- Utility Functions -----------------

def clean_money(s):
    if s is None:
        return None
    s = str(s).strip()
    m = DOLLAR_RE.search(s)
    if m:
        return m.group(0).replace(' ', '')
    # fallback: numbers with commas
    num = re.search(r'\d{1,3}(?:,\d{3})+', s)
    if num:
        return "$" + num.group(0)
    return None

def find_money_near(text, anchor_keywords=None, window_chars=200):
    if anchor_keywords is None:
        anchor_keywords = ANCHOR_KEYWORDS_GENERAL
    text_low = text.lower() if text else ""
    found = []
    for kw in anchor_keywords:
        idx = text_low.find(kw)
        if idx != -1:
            start = max(0, idx - window_chars)
            end = min(len(text), idx + len(kw) + window_chars)
            snippet = text[start:end]
            found += DOLLAR_RE.findall(snippet)
    if not found and text:
        found = DOLLAR_RE.findall(text[:2000])
    # unique
    unique = []
    for m in found:
        mm = m.replace(' ', '')
        if mm not in unique:
            unique.append(mm)
    return unique

def parse_html_tables_for_fee(html_text):
    results = []
    try:
        dfs = pd.read_html(html_text, flavor='bs4')
    except Exception:
        dfs = []
    for df in dfs:
        # Normalize dtypes and column names
        df_cols = [str(c).lower() for c in df.columns.astype(str)]
        headers = " | ".join(df_cols)
        col_idx = None
        for i, col in enumerate(df_cols):
            if 'non' in col and 'virginia' in col:
                col_idx = i; break
            if 'non' in col and 'resident' in col:
                col_idx = i; break
            if 'out' in col and 'state' in col:
                col_idx = i; break
        row_idx = None
        first_col = df.columns[0]
        for r, cell in enumerate(df[first_col].astype(str)):
            if re.search(r'tuition', cell, flags=re.I):
                row_idx = r; break
        if row_idx is not None and col_idx is not None:
            val = str(df.iloc[row_idx, col_idx])
            results.append({'source': 'table', 'value': clean_money(val), 'headers': headers})
        # fallback: pick any cell that looks like money
        if not results:
            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    try:
                        cell = str(df.iat[r, c])
                    except Exception:
                        cell = ''
                    m = clean_money(cell)
                    if m:
                        results.append({'source': 'table-any', 'value': m, 'headers': headers})
    return results

def extract_text_from_pdf(path):
    text = ""
    # try pdfplumber
    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception:
            pass
    # try PyPDF2
    if PdfReader is not None:
        try:
            reader = PdfReader(path)
            for p in reader.pages:
                try:
                    page_text = p.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception:
            pass
    return ""

def find_program_links(soup, base_url):
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(separator=' ', strip=True)
        href_l = href.lower()
        text_l = text.lower()
        if any(k in href_l for k in VCU_KEYWORDS) or any(k in text_l for k in VCU_KEYWORDS):
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.scheme.startswith('http') and 'vcu.edu' in parsed.netloc:
                links.add(full)
    return list(links)

# ----------------- Scanners -----------------

def scan_remote_vcu(start_url, max_pages=40, delay=0.4):
    """Breadth-first crawl within vcu.edu domain, returns dict url -> metadata"""
    start_domain = urlparse(start_url).netloc
    if 'vcu.edu' not in start_domain.lower():
        raise ValueError("Start URL must be within vcu.edu domain")
    to_visit = [start_url]
    visited = set()
    results = {}
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            r.raise_for_status()
            html = r.text
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = soup.get_text(separator='\n', strip=True)
            tables = parse_html_tables_for_fee(html)
            program_links = find_program_links(soup, url)
            results[url] = {'title': title, 'text': text, 'tables': tables, 'program_links': program_links}
            # discover internal links
            for a in soup.find_all('a', href=True):
                href = urljoin(url, a['href'])
                parsed = urlparse(href)
                if parsed.scheme.startswith('http') and 'vcu.edu' in parsed.netloc:
                    if href not in visited and href not in to_visit:
                        to_visit.append(href)
        except Exception as e:
            results[url] = {'title': 'error', 'text': '', 'tables': [], 'program_links': []}
        visited.add(url)
        time.sleep(delay)
    return results

def scan_local_folder(folder_path):
    results = {}
    folder = Path(folder_path)
    if not folder.exists():
        return results
    for file in folder.rglob('*'):
        if not file.is_file():
            continue
        lower = file.suffix.lower()
        try:
            if lower in ['.html', '.htm']:
                html = file.read_text(encoding='utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.title.string.strip() if soup.title and soup.title.string else file.name
                text = soup.get_text(separator='\n', strip=True)
                tables = parse_html_tables_for_fee(html)
                links = find_program_links(soup, f"file://{file.as_posix()}")
                results[str(file)] = {'title': title, 'text': text, 'tables': tables, 'program_links': links}
            elif lower == '.pdf':
                text = extract_text_from_pdf(str(file))
                results[str(file)] = {'title': file.name, 'text': text, 'tables': [], 'program_links': []}
            elif lower in ['.txt', '.md']:
                text = file.read_text(encoding='utf-8', errors='ignore')
                results[str(file)] = {'title': file.name, 'text': text, 'tables': [], 'program_links': []}
        except Exception:
            results[str(file)] = {'title': file.name, 'text': '', 'tables': [], 'program_links': []}
    return results

# ----------------- Topic matching & fee extraction -----------------

def fetch_single_url(url, timeout=15):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text(separator='\n', strip=True)
        tables = parse_html_tables_for_fee(html)
        program_links = find_program_links(soup, url)
        return {url: {'title': title, 'text': text, 'tables': tables, 'program_links': program_links}}
    except Exception:
        return {url: {'title': 'error', 'text': '', 'tables': [], 'program_links': []}}

def score_page_for_topic(page_data, url, topic_tokens):
    title = (page_data.get('title') or "").lower()
    url_lower = url.lower()
    text = (page_data.get('text') or "").lower()
    if not topic_tokens:
        return 0.0
    title_matches = sum(1 for t in topic_tokens if t in title)
    url_matches = sum(1 for t in topic_tokens if t in url_lower)
    text_matches = sum(1 for t in topic_tokens if t in text)
    max_score = 3 * len(topic_tokens) + 2 * len(topic_tokens) + 1 * len(topic_tokens)
    raw = 3 * title_matches + 2 * url_matches + 1 * text_matches
    return raw / max_score if max_score > 0 else 0.0

def find_fees_by_topic(topic, results_dict, top_k_pages=12, anchor_keywords=None):
    if anchor_keywords is None:
        anchor_keywords = ANCHOR_KEYWORDS_GENERAL
    topic_tokens = [t for t in re.findall(r'\b[a-zA-Z]{2,}\b', topic.lower()) if t not in ('the','and','of','for','to','a')]
    scored = []
    for url, data in results_dict.items():
        score = score_page_for_topic(data, url, topic_tokens)
        scored.append((url, data, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    selected = [item for item in scored if item[2] > 0.0][:top_k_pages]
    if not selected and topic_tokens:
        fallback = [item for item in scored if any(t in (item[1].get('text') or "").lower() or t in (item[1].get('title') or "").lower() for t in topic_tokens)]
        selected = fallback[:top_k_pages]
    fee_rows = []
    for url, data, score in selected:
        for t in data.get('tables', []):
            val = t.get('value')
            if val:
                fee_rows.append({
                    'value': val,
                    'url': url,
                    'source': t.get('source', 'table'),
                    'headers': t.get('headers', ''),
                    'snippet': '',
                    'score': score
                })
        text = data.get('text') or ''
        for anchor in anchor_keywords:
            for m in re.finditer(re.escape(anchor), text, flags=re.I):
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                window = text[start:end]
                monies = DOLLAR_RE.findall(window)
                for money in monies:
                    snippet = window.replace('\n',' ')[:300]
                    fee_rows.append({
                        'value': money.replace(' ', ''),
                        'url': url,
                        'source': f'near "{anchor}"',
                        'headers': '',
                        'snippet': snippet,
                        'score': score
                    })
        if not any(fr['url'] == url for fr in fee_rows):
            monies = DOLLAR_RE.findall(text[:2000])
            for money in monies:
                snippet = text[:200].replace('\n',' ')
                fee_rows.append({
                    'value': money.replace(' ', ''),
                    'url': url,
                    'source': 'anywhere',
                    'headers': '',
                    'snippet': snippet,
                    'score': score
                })
    unique = {}
    for r in fee_rows:
        key = (r['value'], r['url'], r['source'])
        if key not in unique or r['score'] > unique[key]['score']:
            unique[key] = r
    df = pd.DataFrame(list(unique.values()))
    if df.empty:
        return df, selected
    df['value_norm'] = df['value'].str.replace(r'[^\d\.,]', '', regex=True).str.replace(',', '')
    freq = df.groupby('value').size().reset_index(name='count')
    df = df.merge(freq, on='value', how='left')
    df = df.sort_values(['count','score'], ascending=[False, False]).reset_index(drop=True)
    return df, selected

# ----------------- Streamlit UI -----------------

st.title("VCU Universal Scraper & Topic Fee Finder")
st.markdown("Crawl VCU pages and/or scan a local folder to extract tuition & fees and program pages. No hardcoded values — general topic matching and ranking.")

# Sidebar: configuration
with st.sidebar:
    st.header("Options")
    crawl_seed = st.text_input("Start URL (VCU domain)", "https://admissions.vcu.edu/cost-aid/tuition-fees/")
    crawl_pages = st.slider("Max pages to crawl", 5, 200, 40)
    crawl_delay = st.number_input("Delay between requests (seconds)", min_value=0.0, max_value=3.0, value=0.4, step=0.1)
    local_folder = st.text_input("Local folder to scan (optional)", r"C:\Users\somna\OneDrive\Fall_Aug_Dec25_MDA\DAPT619NLPAI\VCU_URL_DATA")
    st.markdown("---")
    st.markdown("**PDF parsing:** If `pdfplumber` is installed, PDF text extraction is more robust.")
    st.markdown("Install dependencies: `pip install streamlit requests beautifulsoup4 pandas lxml pdfplumber PyPDF2`")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Crawl VCU (remote)")
    if st.button("🔍 Crawl remote site"):
        if not crawl_seed or 'vcu.edu' not in urlparse(crawl_seed).netloc.lower():
            st.error("Please provide a vcu.edu start URL.")
        else:
            with st.spinner(f"Crawling {crawl_seed} ..."):
                try:
                    remote_results = scan_remote_vcu(crawl_seed, max_pages=crawl_pages, delay=crawl_delay)
                    st.session_state['remote_results'] = remote_results
                    st.success(f"Finished crawling — scanned {len(remote_results)} pages.")
                    # show summary of pages with tables
                    count_tables = sum(1 for v in remote_results.values() if v.get('tables'))
                    st.write(f"Pages with parsed tables: {count_tables}")
                    if count_tables > 0:
                        st.info("Parsed tables usually include tuition & fees tables when present.")
                except Exception as e:
                    st.error(f"Crawl error: {e}")

    if 'remote_results' in st.session_state:
        if st.checkbox("Show crawled page titles", value=False):
            for url, data in list(st.session_state['remote_results'].items())[:50]:
                st.write(f"- **{data.get('title','')}**")
                st.caption(url)

with col2:
    st.subheader("2) Scan local folder")
    if st.button("📂 Scan local folder"):
        folder = local_folder
        if not folder or not os.path.isdir(folder):
            st.error("Please provide a valid folder path.")
        else:
            with st.spinner("Scanning local folder..."):
                local_results = scan_local_folder(folder)
                st.session_state['local_results'] = local_results
                st.success(f"Scanned local folder — found {len(local_results)} files.")
                # quick stats
                pdf_count = sum(1 for k in local_results if str(k).lower().endswith('.pdf'))
                html_count = sum(1 for k in local_results if str(k).lower().endswith(('.html','.htm')))
                st.write(f"PDFs: {pdf_count}, HTML/HTM: {html_count}")

    if 'local_results' in st.session_state:
        if st.checkbox("Show local files scanned", value=False):
            for k, v in list(st.session_state['local_results'].items())[:50]:
                st.write(f"- **{v.get('title','')}**")
                st.caption(k)

st.markdown("---")
st.header("3) Generalized Topic Search (no hardcoding)")

topic_query = st.text_input("Enter a topic or query (examples: academics, tuition, programs, undergraduate)", "")
topic_btn = st.button("Analyze Topic")

if topic_btn:
    # combine remote and local results into a single dictionary
    combined_results = {}
    if 'remote_results' in st.session_state:
        combined_results.update(st.session_state['remote_results'])
    if 'local_results' in st.session_state:
        combined_results.update(st.session_state['local_results'])
    # auto-fetch academics hub if query includes academics/program/course
    if any(k in topic_query.lower() for k in ['academics', 'program', 'course', 'courses']):
        academics_url = 'https://admissions.vcu.edu/academics/'
        if academics_url not in combined_results:
            st.info(f"Auto-fetching {academics_url} to enrich program/academics results...")
            combined_results.update(fetch_single_url(academics_url))
    if not combined_results:
        st.warning("No scanned pages found. Fetching seeds: tuition-fees and academics.")
        seeds = [
            "https://admissions.vcu.edu/cost-aid/tuition-fees/",
            "https://admissions.vcu.edu/academics/"
        ]
        for s in seeds:
            combined_results.update(fetch_single_url(s))
    df_fees, matched_pages = find_fees_by_topic(topic_query or "tuition", combined_results, top_k_pages=20)
    if df_fees.empty:
        st.info("No fee amounts were found for that topic in the scanned pages.")
        if matched_pages:
            st.subheader("Top matched pages (no fees found)")
            for url, data, score in matched_pages[:10]:
                st.write(f"- {data.get('title') or url}  ({score:.2f})")
                st.caption(url)
    else:
        st.subheader("Extracted fees (grouped & ranked)")
        display_cols = ['value','count','score','url','source','snippet']
        st.dataframe(df_fees[display_cols].head(250), height=360)
        top_value = df_fees.iloc[0]['value']
        top_count = int(df_fees.iloc[0]['count'])
        st.success(f"Top fee found: **{top_value}**  — found {top_count} time(s) among matched pages")
        st.subheader(f"Pages containing {top_value}")
        top_rows = df_fees[df_fees['value'] == top_value]
        for _, r in top_rows.iterrows():
            st.write(f"- {r['url']}  —  {r['source']}  (score={r['score']:.2f})")
            if r['snippet']:
                st.caption(r['snippet'][:400] + ("..." if len(r['snippet'])>400 else ""))
        # show program candidates when query is academic-ish
        if any(k in topic_query.lower() for k in ['academics', 'program', 'course', 'courses']):
            st.subheader("Program candidates and discovered program links")
            seen = set()
            for url, data, score in matched_pages[:40]:
                for pl in (data.get('program_links') or []):
                    if pl not in seen:
                        st.write(f"- {pl}")
                        seen.add(pl)
            if not seen:
                st.info("No program links found in matched pages — try a wider crawl or scan local files.")

st.markdown("---")
st.markdown("**Tips:** If the tuition table is rendered by JavaScript, the `requests`-based crawler won't see it. Save the page locally (or save it as PDF) and use the local folder scanner. For robust PDF table parsing consider camelot/tabula; for JS rendering consider Playwright or Selenium.")
