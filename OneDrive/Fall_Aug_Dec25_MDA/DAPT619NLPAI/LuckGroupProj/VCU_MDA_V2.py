# VCU_MDA_V3.py
"""
MDA Weekend — Focused Q&A (V3)
- Prioritizes extraction of 'icon card' tiles (Qualifications, Program Length, etc.)
- If user asks about 'qualifications' it returns the tile content exactly.
- Fall back to summarization & news extraction if needed.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import pandas as pd
import altair as alt
from dateutil import parser as dateparser
from io import BytesIO
from PIL import Image
import time

st.set_page_config(page_title="MDA Weekend — Focused Q&A v3", layout="wide")

# Primary URLs
PRIMARY_BASE = "https://business.vcu.edu/graduate-programs/mda-weekend/"
PRIMARY_NEWS = PRIMARY_BASE + "#newsSection"
PRIMARY_D_EN = PRIMARY_BASE + "#d.en.586297"
PRIMARY_LIST = [PRIMARY_BASE, PRIMARY_NEWS, PRIMARY_D_EN]

# regex
DOLLAR_RE = re.compile(r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
DATE_RE = re.compile(
    r'(?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
    r'Sep(?:t(?:ember)?)|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\w\.\s,]*\d{1,2}(?:,?\s*\d{4})?)',
    flags=re.I
)

# ------------------------ utilities ------------------------

def fetch_url(url, timeout=12):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        return soup, html
    except Exception as e:
        st.warning(f"Fetch error for {url}: {e}")
        return None, None

def get_section_by_id(soup, idname):
    if not soup:
        return "", None
    el = soup.find(id=idname)
    if not el:
        el = soup.find(attrs={"name": idname}) or soup.find(id=lambda x: x and idname in x)
    text = el.get_text(separator="\n", strip=True) if el else ""
    return text, el

def extract_news_items_from_soup(soup, base_url=PRIMARY_BASE):
    if not soup:
        return []
    news_container = soup.find(id='newsSection') or soup.find(class_=re.compile(r'news', flags=re.I))
    if not news_container:
        headers = soup.find_all(re.compile('h[1-6]'))
        for h in headers:
            if 'news' in h.get_text(strip=True).lower():
                sib = h.find_next_sibling()
                if sib:
                    news_container = sib
                    break
    items = []
    if news_container:
        candidates = news_container.find_all(['article','li','div'], recursive=True)
        if not candidates:
            candidates = [news_container]
        for c in candidates:
            title_tag = c.find(['h1','h2','h3','h4','h5'])
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                a = c.find('a')
                title = a.get_text(strip=True) if a else c.get_text(strip=True)[:80]
            date_text = ""
            time_tag = c.find('time')
            if time_tag and time_tag.get_text(strip=True):
                date_text = time_tag.get_text(strip=True)
            else:
                m = DATE_RE.search(c.get_text(" ", strip=True))
                date_text = m.group(0) if m else ""
            p = c.find('p')
            snippet = p.get_text(strip=True) if p else c.get_text(" ", strip=True)[:300]
            img_tag = c.find('img')
            img_url = None
            if img_tag and img_tag.get('src'):
                img_url = urljoin(base_url, img_tag.get('src'))
            link = None
            a = c.find('a', href=True)
            if a:
                link = urljoin(base_url, a['href'])
            items.append({'title':title,'date_text':date_text,'snippet':snippet,'img':img_url,'link':link,'raw_html':str(c)})
    return items

def top_sentences_for_query(text, query, top_k=4):
    if not text:
        return []
    candidates = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    def score_sent(s):
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
        if not words or not tokens:
            return 0
        return len(words & tokens) / len(tokens)
    scored = [(s.strip(), score_sent(s)) for s in candidates if len(s.strip())>20]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s,sc in scored[:top_k]]

def summarize_from_sources(question, primary_texts, fallback_texts=None, max_sentences=3):
    cited=[]
    lines=[]
    for url,txt in primary_texts:
        sents = top_sentences_for_query(txt, question, top_k=max_sentences)
        if sents:
            cited.append(url)
            lines.append(f"From {urlparse(url).netloc} ({url}):")
            for s in sents:
                lines.append("  "+s.strip())
            if len(lines) >= max_sentences + 2:
                break
    if not lines and fallback_texts:
        for url,txt in fallback_texts:
            sents = top_sentences_for_query(txt, question, top_k=max_sentences)
            if sents:
                cited.append(url)
                lines.append(f"From {urlparse(url).netloc} ({url}):")
                for s in sents:
                    lines.append("  "+s.strip())
                if len(lines) >= max_sentences + 2:
                    break
    if not lines:
        answer = ("I couldn't find a direct answer on the MDA Weekend pages. "
                  "Try a different phrase or check the program page and news section directly: "
                  f"{PRIMARY_BASE} (program), {PRIMARY_NEWS} (news).")
        return answer, []
    answer = "\n\n".join(lines)
    return answer, cited

def extract_money_items(text):
    if not text:
        return []
    found = DOLLAR_RE.findall(text)
    uniq=[]
    for f in found:
        v=f.replace(' ','')
        if v not in uniq:
            uniq.append(v)
    return uniq

# ----------------- New: tile extraction -----------------

def extract_icon_tiles(soup):
    """
    Find short icon/feature/tile blocks (Program Length, Qualifications, etc.)
    heuristics: a container (div/section) that includes a small heading and short descriptive text.
    Returns list of {'title','desc','html'} deduped by title.
    """
    tiles = []
    if not soup:
        return tiles
    # gather candidate containers
    # Search for divs/sections that contain a short heading and some short text
    candidates = soup.find_all(['div','section'], recursive=True)
    for c in candidates:
        # find heading inside container
        h = c.find(['h2','h3','h4','h5'])
        if not h:
            continue
        title = h.get_text(strip=True)
        if not title or len(title) > 60:
            continue
        # find a short descriptive text inside same container
        desc = ""
        # first prefer paragraphs inside
        p = c.find('p')
        if p and len(p.get_text(strip=True)) < 200:
            desc = p.get_text(strip=True)
        else:
            # otherwise find spans/divs/strong that look like a short label or number
            small_texts = []
            for tag in c.find_all(['div','span','strong','li']):
                txt = tag.get_text(strip=True)
                if txt and txt.lower() != title.lower() and 0 < len(txt) < 200:
                    small_texts.append(txt)
            # prefer the first short non-empty
            for s in small_texts:
                # ignore if it is the heading again
                if s.lower() != title.lower():
                    desc = s
                    break
        # sometimes the tile is structured as header + small boxes; include if desc found
        tiles.append({'title': title, 'desc': desc.strip(), 'html': str(c)})
    # dedupe by title (case-insensitive)
    unique = {}
    for t in tiles:
        key = t['title'].strip().lower()
        if key not in unique:
            unique[key] = t
        else:
            # keep the one with description if other missing
            if not unique[key]['desc'] and t['desc']:
                unique[key] = t
    # return list
    return list(unique.values())

# ----------------- cache primary pages -----------------

if 'primary_cache' not in st.session_state:
    st.session_state.primary_cache = {}

def ensure_primary_pages():
    cache = st.session_state.primary_cache
    for url in PRIMARY_LIST:
        if url not in cache:
            soup, html = fetch_url(url)
            cache[url] = (soup, html)
            time.sleep(0.2)
    return cache

# ----------------- UI -----------------

st.title("MDA Weekend — Focused Q&A v3")
st.markdown("This assistant prioritizes the MDA Weekend program tiles (Qualifications etc.). Ask `What are the qualifications?` and it will return the tile content directly.")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Load MDA pages"):
        ensure_primary_pages()
        st.success("Loaded MDA program pages into cache.")
with col2:
    if st.button("Show raw MDA page text"):
        cache = ensure_primary_pages()
        for url,(soup,html) in cache.items():
            st.subheader(url)
            if soup:
                st.text(soup.get_text("\n", strip=True)[:800])
            else:
                st.text("[no content]")
with col3:
    if st.button("Refresh MDA pages (re-fetch)"):
        st.session_state.primary_cache = {}
        ensure_primary_pages()
        st.success("Refreshed cache.")

st.markdown("---")
query = st.text_input("Ask a question about MDA Weekend (news, program, qualifications, fees, schedule):", "")
submitted = st.button("Answer")

with st.sidebar:
    st.header("Primary sources")
    st.write(f"- {PRIMARY_BASE}")
    st.write(f"- {PRIMARY_NEWS}")
    st.write(f"- {PRIMARY_D_EN}")
    st.markdown("---")
    st.write("Tip: Try short queries like `qualifications` or `What are the qualifications?`")

if submitted:
    cache = ensure_primary_pages()
    base_soup, base_html = cache.get(PRIMARY_BASE, (None,None))
    det_text, det_el = get_section_by_id(base_soup, 'd.en.586297')
    ntext, n_el = get_section_by_id(base_soup, 'newsSection')
    full_text = base_soup.get_text("\n", strip=True) if base_soup else ""

    # Extract tiles
    tiles = extract_icon_tiles(base_soup)
    # make map from lower-title -> desc
    tile_map = {t['title'].strip().lower(): t['desc'] for t in tiles}

    q = (query or "").strip()
    qlow = q.lower()

    # If user asks exactly about a tile name (qualifications, program length, application deadlines ...)
    # or query is short (<=3 words) and matches a tile
    def find_tile_answer(qtext):
        # direct match by containing keywords
        for title_lower, desc in tile_map.items():
            if title_lower and any(k in qtext for k in [title_lower, title_lower.split()[0]]):
                return title_lower, desc
        # try fuzzy containment: e.g., query 'qualification' -> title 'Qualifications'
        for title_lower, desc in tile_map.items():
            if 'qualification' in qtext and 'qualif' in title_lower:
                return title_lower, desc
            if 'program length' in qtext and 'program length' in title_lower:
                return title_lower, desc
            if 'application dead' in qtext and 'application' in title_lower:
                return title_lower, desc
        return None, None

    title_match, desc_match = find_tile_answer(qlow)

    if title_match and (len(qlow.split()) <= 4 or 'qualification' in qlow or 'program length' in qlow or 'application' in qlow):
        # Return tile content verbatim (this fixes the Qualifications answer)
        st.subheader(f"{title_match.title()}")
        if desc_match:
            st.write(desc_match)
        else:
            st.info("Tile found but no short description extracted. Showing full tile HTML for inspection:")
            # show tile html for debugging
            for t in tiles:
                if t['title'].strip().lower() == title_match:
                    st.code(t['html'][:2000])
                    break
        # also show other tiles in a short table
        if tiles:
            st.markdown("---")
            st.subheader("Other program tiles found")
            rows = []
            for t in tiles:
                rows.append({'title': t['title'], 'desc': t['desc']})
            df = pd.DataFrame(rows)
            st.dataframe(df, height=220)
        st.stop()

    # If not matched to tile: fall back to previous logic
    is_event_query = any(w in qlow for w in ["event","events","calendar","oct","nov","date","when","time","news"])
    is_fee_query = any(w in qlow for w in ["tuition","fee","fees","cost","non-virginia","out-of-state","resident"])
    is_program_query = any(w in qlow for w in ["curriculum","learning","outcome","course","program","admission","apply","schedule","qualif"])

    # Events / News
    if is_event_query:
        # parse simple date range
        start_dt,end_dt = None,None
        try:
            m_to = re.search(r'(.+?)\s+(?:to|-|—)\s+(.+)', q)
            if m_to:
                a = dateparser.parse(m_to.group(1), fuzzy=True)
                b = dateparser.parse(m_to.group(2), fuzzy=True)
                if a and b:
                    start_dt,end_dt=a,b
            else:
                m = re.search(r'([A-Za-z]{3,9})\s*(\d{4})', q)
                if m:
                    mon = m.group(1); yr = int(m.group(2))
                    start_dt = dateparser.parse(f"{mon} 1 {yr}")
                    end_dt = dateparser.parse(f"{mon} 28 {yr}")
        except Exception:
            start_dt,end_dt=None,None

        news_items = extract_news_items_from_soup(base_soup, base_url=PRIMARY_BASE)
        def parse_date_safe(dt_text):
            if not dt_text:
                return None
            try:
                return dateparser.parse(dt_text, fuzzy=True)
            except Exception:
                return None
        if start_dt and end_dt:
            filtered=[]
            for it in news_items:
                dt = parse_date_safe(it.get('date_text') or "")
                if dt and dt.date() >= start_dt.date() and dt.date() <= end_dt.date():
                    filtered.append(it)
            news_items = filtered
        if not news_items:
            st.info("No matching news/events found.")
        else:
            st.subheader("News / Events")
            st.write(f"Found {len(news_items)} items. Highlights:")
            for it in news_items[:6]:
                st.write(f"- **{it['title']}** — {it['date_text']}")
                st.caption(it['snippet'][:300])
            # show table
            rows=[]
            for it in news_items:
                dt = parse_date_safe(it.get('date_text') or "")
                rows.append({'date': dt.date().isoformat() if dt else it.get('date_text'),'title':it['title'],'snippet':it['snippet'],'link':it['link'],'img':it['img']})
            df = pd.DataFrame(rows)
            st.dataframe(df[['date','title','snippet','link']], height=300)
            # timeline
            df_chart = df.dropna(subset=['date']).copy()
            if not df_chart.empty:
                try:
                    df_chart['date_parsed'] = pd.to_datetime(df_chart['date'])
                    chart = alt.Chart(df_chart).mark_circle(size=120).encode(
                        x=alt.X('date_parsed:T', title='Date'),
                        y=alt.Y('title:N', title='Event', sort=None),
                        tooltip=['title','date_parsed','snippet','link']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass
        st.stop()

    # Fees or program queries
    if is_fee_query or is_program_query:
        primary_texts=[]
        if det_text:
            primary_texts.append((PRIMARY_D_EN, det_text))
        if ntext:
            primary_texts.append((PRIMARY_NEWS, ntext))
        primary_texts.append((PRIMARY_BASE, full_text))

        ans, cited = summarize_from_sources(q, primary_texts, fallback_texts=[(PRIMARY_BASE, full_text)], max_sentences=4)
        if is_fee_query:
            money = extract_money_items(det_text) + extract_money_items(full_text)
            money = list(dict.fromkeys(money))
            if money:
                money_lines = [f"I found these fee amounts on the MDA Weekend pages: {', '.join(money)}."]
                ans = "\n\n".join([ans] + money_lines)
        st.subheader("Answer")
        st.text(ans)
        if cited:
            st.caption("Cited: " + ", ".join(cited))
        # give tiles as extra structured info
        if tiles:
            st.markdown("---")
            st.subheader("Program tiles (extracted)")
            rows=[]
            for t in tiles:
                rows.append({'title':t['title'],'desc':t['desc']})
            st.dataframe(pd.DataFrame(rows), height=200)
        st.stop()

    # fallback general
    prims=[(PRIMARY_BASE, full_text)]
    ans,cited = summarize_from_sources(q, prims, max_sentences=4)
    st.subheader("Answer (general)")
    st.text(ans)
    if cited:
        st.caption("Cited: " + ", ".join(cited))

else:
    # preview
    st.subheader("MDA Weekend — quick preview")
    cache = ensure_primary_pages()
    base_soup, base_html = cache.get(PRIMARY_BASE, (None,None))
    if base_soup:
        h1 = base_soup.find('h1')
        intro = h1.get_text(strip=True) if h1 else "MDA Weekend"
        det_text, det_el = get_section_by_id(base_soup, 'd.en.586297')
        if not det_text:
            headings = base_soup.find_all(['h2','h3','h4'])
            det_text = ""
            for h in headings:
                txt = h.get_text(strip=True)
                if txt and ('overview' in txt.lower() or 'program' in txt.lower()):
                    p = h.find_next('p')
                    if p:
                        det_text = p.get_text(strip=True)
                        break
        st.markdown(f"**{intro}**")
        if det_text:
            st.write(det_text[:800] + ("..." if len(det_text) > 800 else ""))
        else:
            st.write("Program detail not detected automatically. Click 'Load MDA pages' to fetch full page.")
    else:
        st.info("Unable to fetch MDA program page. Click 'Load MDA pages' to fetch.")

    st.markdown("---")
    st.subheader("News preview")
    news_list = extract_news_items_from_soup(base_soup, base_url=PRIMARY_BASE) if base_soup else []
    if news_list:
        for it in news_list[:6]:
            st.markdown(f"**{it.get('title')}** — {it.get('date_text')}")
            st.caption(it.get('snippet')[:200])
    else:
        st.info("No news section auto-detected on the page preview. Click 'Load MDA pages' to fetch live content.")

st.markdown("---")
st.markdown("If you'd like: I can make the tile extraction stricter, or add Playwright rendering to ensure we capture dynamically rendered content.")
