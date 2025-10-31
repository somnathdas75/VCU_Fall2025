# VCU_MDA_V5_RAG.py
"""
MDA Weekend — Concise RAG Chatbot (V5)
- Provides short, direct answers
- Uses Chroma DB for chat history
- Maintains tile extraction for key program details
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import pandas as pd
import time
import hashlib
import json
from datetime import datetime

# Try to import Chroma DB (optional)
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.warning("Chroma DB not available. Install with: pip install chromadb")

st.set_page_config(page_title="MDA Weekend — Concise Chat", layout="wide")

# Primary URLs
PRIMARY_BASE = "https://business.vcu.edu/graduate-programs/mda-weekend/"
PRIMARY_LIST = [PRIMARY_BASE]

# ------------------------ Chroma DB Manager ------------------------

class ChromaDBChatHistory:
    def __init__(self):
        self.client = None
        self.collection = None
        self.initialize_db()
    
    def initialize_db(self):
        if not CHROMA_AVAILABLE:
            return
            
        try:
            self.client = chromadb.Client()
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection("mda_chat_history")
            except:
                self.collection = self.client.create_collection(
                    name="mda_chat_history",
                    metadata={"description": "MDA Weekend program chat history"}
                )
        except Exception as e:
            st.warning(f"Chroma DB initialization failed: {e}")
    
    def add_message(self, query, response, sources=None):
        if not self.collection:
            return
            
        try:
            message_id = f"msg_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
            self.collection.add(
                documents=[response],
                metadatas=[{
                    "query": query,
                    "response": response,
                    "sources": json.dumps(sources or []),
                    "timestamp": datetime.now().isoformat(),
                    "type": "chat_message"
                }],
                ids=[message_id]
            )
        except Exception as e:
            st.warning(f"Failed to save to Chroma DB: {e}")
    
    def get_recent_history(self, limit=10):
        if not self.collection:
            return []
            
        try:
            results = self.collection.get(limit=limit)
            history = []
            for i in range(len(results['ids'])):
                history.append({
                    'query': results['metadatas'][i]['query'],
                    'response': results['metadatas'][i]['response'],
                    'sources': json.loads(results['metadatas'][i]['sources']),
                    'timestamp': results['metadatas'][i]['timestamp']
                })
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            st.warning(f"Failed to load from Chroma DB: {e}")
            return []

# ------------------------ Concise Response Generator ------------------------

class ConciseResponseGenerator:
    def __init__(self):
        self.program_data = self.load_program_data()
    
    def load_program_data(self):
        """Pre-defined concise program information"""
        return {
            "application_deadlines": {
                "answer": "Application Deadlines\n\nRound 1: November 1\nRound 2: March 1\nRound 3: May 1\nRound 4: July 1",
                "sources": [PRIMARY_BASE]
            },
            "qualifications": {
                "answer": "Qualifications\n\n• Bachelor's degree\n• 2+ years of professional work experience\n• GRE/GMAT not required",
                "sources": [PRIMARY_BASE]
            },
            "program_length": {
                "answer": "Program Length: 16-20 months",
                "sources": [PRIMARY_BASE]
            },
            "cost": {
                "answer": "Total Cost: $45,000\n\nIncludes tuition, fees, meals, parking, and course materials",
                "sources": [PRIMARY_BASE]
            },
            "format": {
                "answer": "Program Format\n\n• Hybrid weekend classes\n• Two weekends per month\n• One mandatory in-person, one HyFlex",
                "sources": [PRIMARY_BASE]
            },
            "curriculum": {
                "answer": "Curriculum\n\n• 33-credit fixed curriculum\n• Technical tools: Python, R, SQL, Tableau, Power BI\n• 12-month hands-on practicum",
                "sources": [PRIMARY_BASE]
            },
            "application_process": {
                "answer": "Application Requirements\n\n• Online application ($75 fee)\n• Transcripts\n• 1-2 page statement of intent\n• Current resume\n• 3 letters of recommendation",
                "sources": [PRIMARY_BASE]
            }
        }
    
    def get_concise_answer(self, query):
        """Get short, direct answer based on query"""
        query_lower = query.lower()
        
        # Direct pattern matching for common questions
        patterns = {
            'deadline': 'application_deadlines',
            'application date': 'application_deadlines', 
            'when to apply': 'application_deadlines',
            'dateline': 'application_deadlines',
            'qualification': 'qualifications',
            'requirement': 'qualifications',
            'prerequisite': 'qualifications',
            'how long': 'program_length',
            'duration': 'program_length',
            'length': 'program_length',
            'cost': 'cost',
            'tuition': 'cost',
            'fee': 'cost',
            'price': 'cost',
            'format': 'format',
            'schedule': 'format',
            'class': 'format',
            'curriculum': 'curriculum',
            'course': 'curriculum',
            'what learn': 'curriculum',
            'apply': 'application_process',
            'how apply': 'application_process',
            'requirement': 'application_process'
        }
        
        for pattern, data_key in patterns.items():
            if pattern in query_lower:
                return self.program_data[data_key]
        
        # Fallback for unknown questions
        return {
            "answer": "I can provide concise information about:\n• Application deadlines\n• Qualifications\n• Program length\n• Costs\n• Curriculum\n• Application process\n\nPlease ask about any of these topics!",
            "sources": [PRIMARY_BASE]
        }

# ------------------------ Enhanced Tile Extraction ------------------------

def fetch_url(url, timeout=10):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        return None

def extract_icon_tiles(soup):
    """Extract key program information tiles"""
    tiles = []
    if not soup:
        return tiles
    
    # Look for common tile patterns
    tile_selectors = [
        'div[class*="card"]',
        'div[class*="tile"]',
        'div[class*="feature"]',
        'div[class*="stat"]',
        '.vcualert'
    ]
    
    for selector in tile_selectors:
        elements = soup.select(selector)
        for elem in elements:
            title_elem = elem.find(['h2', 'h3', 'h4', 'h5', 'strong'])
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) < 50:
                    # Get description
                    desc = ""
                    p_elem = elem.find('p')
                    if p_elem:
                        desc = p_elem.get_text(strip=True)
                    else:
                        # Get any text content
                        text = elem.get_text(" ", strip=True)
                        desc = text.replace(title, "").strip()[:100]
                    
                    tiles.append({
                        'title': title,
                        'desc': desc
                    })
    
    return tiles[:10]  # Return top 10 tiles

# ------------------------ Streamlit UI ------------------------

def main():
    st.title("💬 MDA Weekend Assistant")
    st.markdown("Get concise answers about the VCU MDA Weekend program")
    
    # Initialize components
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChromaDBChatHistory()
    if 'response_generator' not in st.session_state:
        st.session_state.response_generator = ConciseResponseGenerator()
    if 'local_chat_history' not in st.session_state:
        st.session_state.local_chat_history = []
    if 'program_tiles' not in st.session_state:
        st.session_state.program_tiles = []
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        if st.button("🔄 Load Program Data"):
            with st.spinner("Fetching latest program information..."):
                soup = fetch_url(PRIMARY_BASE)
                if soup:
                    st.session_state.program_tiles = extract_icon_tiles(soup)
                    st.success(f"Loaded {len(st.session_state.program_tiles)} program details")
                else:
                    st.error("Failed to load program data")
        
        st.markdown("---")
        st.header("📋 Quick Info")
        
        # Quick action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗓️ Deadlines"):
                response = st.session_state.response_generator.get_concise_answer("deadlines")
                st.session_state.local_chat_history.append({
                    "role": "user", 
                    "content": "Application deadlines"
                })
                st.session_state.local_chat_history.append({
                    "role": "assistant", 
                    "content": response["answer"]
                })
        with col2:
            if st.button("💰 Cost"):
                response = st.session_state.response_generator.get_concise_answer("cost")
                st.session_state.local_chat_history.append({
                    "role": "user", 
                    "content": "Program cost"
                })
                st.session_state.local_chat_history.append({
                    "role": "assistant", 
                    "content": response["answer"]
                })
        
        st.markdown("---")
        st.header("💡 Common Questions")
        questions = [
            "What are the application deadlines?",
            "What are the qualifications?", 
            "How long is the program?",
            "What is the total cost?",
            "What is the program format?",
            "What is the curriculum?"
        ]
        for q in questions:
            if st.button(f"• {q}", key=f"q_{q}"):
                st.session_state.local_chat_history.append({
                    "role": "user", 
                    "content": q
                })
                response = st.session_state.response_generator.get_concise_answer(q)
                st.session_state.local_chat_history.append({
                    "role": "assistant", 
                    "content": response["answer"]
                })
                # Save to Chroma DB
                st.session_state.chat_manager.add_message(q, response["answer"], response["sources"])
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.local_chat_history = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat")
        
        # Display chat history
        for message in st.session_state.local_chat_history[-8:]:  # Last 8 messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        query = st.chat_input("Ask about MDA Weekend program...")
        
        if query:
            # Add user message
            st.session_state.local_chat_history.append({
                "role": "user", 
                "content": query
            })
            
            # Generate and display response
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                response = st.session_state.response_generator.get_concise_answer(query)
                st.write(response["answer"])
                
                # Show sources on expand
                with st.expander("View Sources"):
                    for source in response["sources"]:
                        st.write(f"• {source}")
            
            # Add to chat history
            st.session_state.local_chat_history.append({
                "role": "assistant", 
                "content": response["answer"]
            })
            
            # Save to Chroma DB
            st.session_state.chat_manager.add_message(query, response["answer"], response["sources"])
    
    with col2:
        st.subheader("🎯 Program Details")
        
        if st.session_state.program_tiles:
            for tile in st.session_state.program_tiles[:6]:
                with st.container():
                    st.write(f"**{tile['title']}**")
                    if tile['desc']:
                        st.caption(tile['desc'][:80] + "..." if len(tile['desc']) > 80 else tile['desc'])
                    st.markdown("---")
        else:
            st.info("Click 'Load Program Data' to see key program details")
            
        st.markdown("---")
        st.subheader("📊 Chat Info")
        st.write(f"Messages: {len(st.session_state.local_chat_history)//2}")
        if CHROMA_AVAILABLE:
            st.success("✓ Chroma DB Active")
        else:
            st.warning("✗ Chroma DB Not Available")

if __name__ == "__main__":
    main()