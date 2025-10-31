# VCU_MDA_V5_NLP_AI_LUCKGRP3.py
"""
MDA Weekend — AI-Powered Chatbot (V5)
- Uses spaCy for advanced NLP processing
- Implements semantic similarity with sentence transformers
- Maintains Chroma DB with AI-enhanced embeddings
- OpenAI API functionality commented out to prevent errors
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# AI and NLP imports
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English model with better error handling
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Try to download the model automatically
        try:
            import subprocess
            import sys
            st.info("Downloading spaCy English model... This may take a few minutes.")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            st.success("spaCy English model downloaded successfully!")
        except Exception as e:
            st.warning(f"Could not download spaCy model: {e}")
            st.info("Please run manually: python -m spacy download en_core_web_sm")
            SPACY_AVAILABLE = False
            # Create a fallback NLP processor
            nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    st.warning("spaCy not available. Install with: pip install spacy")
    nlp = None

# Comment out OpenAI functionality to prevent errors
OPENAI_AVAILABLE = False  # Force disable OpenAI

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
    model = None

# Fix ChromaDB compatibility issues
try:
    # First, ensure we're using compatible numpy version
    import pkg_resources
    required = {'numpy'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if not missing:
        # Check numpy version and warn if incompatible
        import numpy as np
        numpy_version = np.__version__
        if numpy_version.startswith('2.'):
            st.warning(f"NumPy {numpy_version} may have compatibility issues. Consider: pip install 'numpy<2.0'")
    
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.warning("Chroma DB not available. Install with: pip install chromadb")
except Exception as e:
    CHROMA_AVAILABLE = False
    st.warning(f"Chroma DB initialization failed: {e}")

st.set_page_config(page_title="MDA Weekend — AI Chatbot", layout="wide")

# Enhanced Primary URLs
PRIMARY_BASE = "https://business.vcu.edu/graduate-programs/mda-weekend/"
BULLETIN_URL = "https://bulletin.vcu.edu/graduate/school-business/decision-analytics-programs/decision-analytics-mda-pro/"
APPLY_URL = "https://gradadmissions.vcu.edu/portal/apply?_gl=1*qxhpby*_gcl_au*NzE4MTA2NDk1LjE3NjE3ODc2NTE.*_ga*Mjk1NjQ2NzY5LjE3NTk3NjQ3MTg.*_ga_WMHV0FXMBD*czE3NjE4NjAxNDEkbzEkZzEkdDE3NjE4NjAzMzAkajYwJGwwJGgw"
REFERRAL_URL = "https://business.vcu.edu/graduate-programs/mda-weekend/referral-award/"

PRIMARY_LIST = [PRIMARY_BASE, BULLETIN_URL]

# ------------------------ AI-Powered NLP Processor ------------------------

class AINLPProcessor:
    def __init__(self):
        self.entity_categories = {
            'program_terms': ['mda', 'weekend', 'decision analytics', 'stem', 'cohort', 'hyflex'],
            'admission_terms': ['deadline', 'apply', 'application', 'qualifications', 'requirements', 'prerequisites'],
            'curriculum_terms': ['curriculum', 'courses', 'credits', 'practicum', 'python', 'sql', 'tableau'],
            'financial_terms': ['cost', 'tuition', 'fee', 'scholarship', 'financial aid', 'referral award']
        }
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not SPACY_AVAILABLE or not text or nlp is None:
            return []
        
        try:
            doc = nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            return entities
        except Exception as e:
            st.warning(f"Error in entity extraction: {e}")
            return []
    
    def analyze_intent(self, query):
        """Analyze user intent using NLP"""
        query_lower = query.lower()
        
        intents = {
            'admission': 0,
            'curriculum': 0,
            'financial': 0,
            'deadline': 0,
            'general_info': 0
        }
        
        # Intent scoring based on keyword presence and patterns
        admission_keywords = ['apply', 'application', 'admission', 'qualification', 'requirement', 'prerequisite']
        curriculum_keywords = ['course', 'curriculum', 'learn', 'study', 'class', 'credit']
        financial_keywords = ['cost', 'tuition', 'fee', 'price', 'scholarship', 'financial']
        deadline_keywords = ['deadline', 'when to apply', 'dateline', 'due date']
        
        for word in admission_keywords:
            if word in query_lower:
                intents['admission'] += 1
        
        for word in curriculum_keywords:
            if word in query_lower:
                intents['curriculum'] += 1
                
        for word in financial_keywords:
            if word in query_lower:
                intents['financial'] += 1
                
        for word in deadline_keywords:
            if word in query_lower:
                intents['deadline'] += 1
        
        # If no specific intent detected, classify as general
        if sum(intents.values()) == 0:
            intents['general_info'] = 1
        
        # Return primary intent
        primary_intent = max(intents.items(), key=lambda x: x[1])
        return primary_intent[0] if primary_intent[1] > 0 else 'general_info'
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or model is None:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0
            return len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Use sentence transformers for better similarity
        try:
            embeddings = model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        except Exception as e:
            st.warning(f"Error in semantic similarity: {e}")
            # Fallback to word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0
            return len(words1.intersection(words2)) / len(words1.union(words2))

# ------------------------ AI Response Generator ------------------------

class AIResponseGenerator:
    def __init__(self):
        self.nlp_processor = AINLPProcessor()
        self.knowledge_base = self.build_knowledge_base()
    
    def build_knowledge_base(self):
        """Build a comprehensive knowledge base with semantic chunks"""
        return {
            "admission": {
                "deadlines": "Application Deadlines: Round 1: November 1, Round 2: March 1, Round 3: May 1, Round 4: July 1",
                "requirements": "Admission requires a bachelor's degree and 2+ years of professional work experience. No GMAT/GRE required.",
                "process": "Submit online application with $75 fee, transcripts, statement of intent, resume, and three recommendation letters.",
                "qualifications": "Qualifications include bachelor's degree, 2+ years work experience, and introductory statistics knowledge."
            },
            "curriculum": {
                "overview": "33-credit fixed curriculum covering Python, R, SQL, Tableau, Power BI, machine learning, and AI.",
                "practicum": "12-month hands-on practicum working with real companies on analytics projects.",
                "tools": "Technical tools include Python, R, SQL, Tableau, Power BI, and various analytics frameworks.",
                "duration": "Program length is 20 months with hybrid weekend classes."
            },
            "financial": {
                "cost": "Total cost: $45,000 including tuition, fees, meals, parking, and course materials.",
                "scholarships": "Alumni referral award available: $2,000 for students referred by VCU alumni.",
                "payment": "Payment plans and financial aid options available for qualified students."
            },
            "program": {
                "format": "Hybrid weekend format: two weekends per month (one in-person, one HyFlex).",
                "cohort": "Cohort-based program with 25-30 students per class.",
                "stem": "STEM-designated program focusing on data analytics and decision sciences."
            }
        }
    
    def generate_ai_response(self, query, context_chunks):
        """Generate intelligent response using AI and context"""
        
        # Analyze user intent
        intent = self.nlp_processor.analyze_intent(query)
        
        # Find most relevant context
        best_context = self.find_most_relevant_context(query, context_chunks)
        
        # Generate response based on intent and context
        return self.generate_rule_based_response(query, intent, best_context)
    
    def find_most_relevant_context(self, query, context_chunks):
        """Find most relevant context using semantic similarity"""
        if not context_chunks:
            return ""
        
        best_similarity = 0
        best_context = ""
        
        for chunk in context_chunks:
            similarity = self.nlp_processor.semantic_similarity(query, chunk)
            if similarity > best_similarity:
                best_similarity = similarity
                best_context = chunk
        
        return best_context if best_similarity > 0.3 else ""
    
    def generate_rule_based_response(self, query, intent, context):
        """Generate response using rule-based approach with NLP enhancements"""
        
        query_lower = query.lower()
        
        # Enhanced response templates with AI-like variations
        responses = {
            'admission': {
                'deadlines': "🗓️ **Application Deadlines**\n\n• Round 1: November 1\n• Round 2: March 1\n• Round 3: May 1\n• Round 4: July 1\n\nI recommend applying early for best consideration!",
                'requirements': "🎓 **Admission Requirements**\n\n• Bachelor's degree from accredited institution\n• 2+ years of professional work experience\n• No GMAT or GRE required\n• Introductory statistics knowledge (20-hour online course available if needed)",
                'process': "📝 **Application Process**\n\n1. Complete online application ($75 fee)\n2. Submit all college transcripts\n3. Provide 1-2 page statement of intent\n4. Upload current resume\n5. Arrange three recommendation letters\n6. International students: English proficiency scores"
            },
            'curriculum': {
                'default': "📚 **MDA Weekend Curriculum**\n\n• 33-credit fixed curriculum (no electives)\n• Technical tools: Python, R, SQL, Tableau, Power BI\n• Advanced topics: Machine Learning, AI, Statistical Modeling\n• 12-month hands-on practicum with real companies\n• Program duration: 20 months"
            },
            'financial': {
                'cost': "💰 **Program Cost: $45,000**\n\nThis comprehensive fee includes:\n• Tuition and university fees\n• Textbooks and software licenses\n• Parking and meals on class weekends\n• Domestic analytics conference attendance\n• Networking and social events",
                'scholarships': "🏆 **Financial Opportunities**\n\n• Alumni Referral Award: $2,000 for referred students\n• Payment plans available\n• Corporate sponsorship options\n• Financial aid for qualified applicants"
            },
            'program': {
                'default': "🎯 **MDA Weekend Program**\n\n• Format: Hybrid weekend classes\n• Schedule: Two weekends per month\n• Delivery: One in-person, one HyFlex weekend\n• Class size: 25-30 students (cohort-based)\n• STEM-designated program"
            }
        }
        
        # Intent-based response selection
        if intent in responses:
            intent_responses = responses[intent]
            
            # Find the most specific response
            for key in intent_responses:
                if key in query_lower:
                    return intent_responses[key]
            
            # Return default for this intent
            return intent_responses.get('default', "I can help with that! Please check our website for detailed information.")
        
        # Fallback response with context
        if context:
            return f"Based on the program information:\n\n{context}\n\nIs there anything specific you'd like to know more about?"
        else:
            return "I can help you with information about admissions, curriculum, costs, and program details. What would you like to know?"

# ------------------------ AI-Enhanced Chroma DB ------------------------

class AIChromaDBChatHistory:
    def __init__(self):
        self.client = None
        self.collection = None
        self.nlp_processor = AINLPProcessor()
        self.initialize_db()
    
    def initialize_db(self):
        if not CHROMA_AVAILABLE:
            return
            
        try:
            self.client = chromadb.Client()
            try:
                self.collection = self.client.get_collection("mda_ai_chat_history")
            except:
                self.collection = self.client.create_collection(
                    name="mda_ai_chat_history",
                    metadata={"description": "AI-enhanced MDA program chat history"}
                )
        except Exception as e:
            st.warning(f"Chroma DB initialization failed: {e}")
    
    def add_message(self, query, response, sources=None, intent=None):
        if not self.collection:
            return
            
        try:
            message_id = f"ai_msg_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
            
            # Enhanced metadata with NLP analysis
            metadata = {
                "query": query,
                "response": response,
                "sources": json.dumps(sources or []),
                "timestamp": datetime.now().isoformat(),
                "intent": intent or "unknown",
                "type": "ai_chat_message"
            }
            
            self.collection.add(
                documents=[response],
                metadatas=[metadata],
                ids=[message_id]
            )
        except Exception as e:
            st.warning(f"Failed to save to Chroma DB: {e}")
    
    def get_similar_conversations(self, query, limit=3):
        """Find similar past conversations using semantic search"""
        if not self.collection:
            return []
            
        try:
            # Get all conversations and compute similarity
            results = self.collection.get()
            similar_conversations = []
            
            for i in range(len(results['ids'])):
                past_query = results['metadatas'][i]['query']
                similarity = self.nlp_processor.semantic_similarity(query, past_query)
                
                if similarity > 0.6:  # Similarity threshold
                    similar_conversations.append({
                        'query': past_query,
                        'response': results['metadatas'][i]['response'],
                        'similarity': similarity
                    })
            
            # Return most similar conversations
            similar_conversations.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_conversations[:limit]
            
        except Exception as e:
            st.warning(f"Failed to find similar conversations: {e}")
            return []

# ------------------------ Enhanced Content Fetcher ------------------------

class AIContentFetcher:
    def __init__(self):
        self.cache = {}
        self.nlp_processor = AINLPProcessor()
    
    def fetch_and_process_content(self, url):
        """Fetch URL content and process with NLP"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Process with NLP - use fallback if spaCy not available
            entities = []
            sentences = []
            
            if SPACY_AVAILABLE and nlp is not None:
                try:
                    entities = self.nlp_processor.extract_entities(text_content)
                    doc = nlp(text_content)
                    sentences = [sent.text for sent in doc.sents][:50]
                except Exception as e:
                    st.warning(f"spaCy processing failed: {e}")
                    sentences = text_content.split('.')[:50]
            else:
                sentences = text_content.split('.')[:50]
            
            return {
                'content': text_content,
                'entities': entities,
                'sentences': sentences,
                'url': url
            }
            
        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
            return None

# ------------------------ Streamlit UI ------------------------

def main():
    st.title("🤖 MDA Weekend — AI-Powered Assistant")
    st.markdown("""
    **Intelligent chatbot with advanced NLP capabilities**
    - Semantic understanding of your questions
    - AI-generated responses
    - Conversation memory and context awareness
    - OpenAI API functionality currently disabled
    """)
    
    # Show installation instructions if needed
    if not SPACY_AVAILABLE:
        with st.expander("🔧 Installation Instructions", expanded=True):
            st.write("""
            **To enable full AI capabilities, please install the missing dependencies:**
            
            ```bash
            # Install spaCy and download English model
            pip install spacy
            python -m spacy download en_core_web_sm
            
            # Install sentence transformers for semantic search
            pip install sentence-transformers
            
            # If ChromaDB fails, try installing compatible numpy:
            pip install "numpy<2.0"
            ```
            """)
    
    # Initialize AI components
    if 'ai_chat_manager' not in st.session_state:
        st.session_state.ai_chat_manager = AIChromaDBChatHistory()
    if 'ai_response_generator' not in st.session_state:
        st.session_state.ai_response_generator = AIResponseGenerator()
    if 'ai_content_fetcher' not in st.session_state:
        st.session_state.ai_content_fetcher = AIContentFetcher()
    if 'ai_chat_history' not in st.session_state:
        st.session_state.ai_chat_history = []
    if 'ai_context' not in st.session_state:
        st.session_state.ai_context = []
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            ai_enabled = st.checkbox("Enable AI", value=True, help="Use AI for intelligent responses")
            nlp_enabled = st.checkbox("Enable NLP", value=SPACY_AVAILABLE, help="Use advanced NLP processing", disabled=not SPACY_AVAILABLE)
        
        with col2:
            semantic_search = st.checkbox("Semantic Search", value=SENTENCE_TRANSFORMERS_AVAILABLE, help="Use semantic similarity", disabled=not SENTENCE_TRANSFORMERS_AVAILABLE)
            show_analysis = st.checkbox("Show Analysis", value=False, help="Show NLP analysis details")
        
        st.markdown("---")
        st.header("🔧 Actions")
        
        if st.button("🔄 Load AI Context"):
            with st.spinner("Loading and processing content with AI..."):
                for url in PRIMARY_LIST:
                    content = st.session_state.ai_content_fetcher.fetch_and_process_content(url)
                    if content:
                        st.session_state.ai_context.extend(content['sentences'])
                st.success(f"Loaded {len(st.session_state.ai_context)} context sentences")
        
        if st.button("🧠 Analyze Capabilities"):
            col1, col2, col3 = st.columns(3)
            with col1:
                status = "✅" if SPACY_AVAILABLE else "❌"
                st.metric("NLP", status)
            with col2:
                st.metric("AI", "❌")  # OpenAI disabled
            with col3:
                status = "✅" if SENTENCE_TRANSFORMERS_AVAILABLE else "❌"
                st.metric("Semantic", status)
            
            if not SPACY_AVAILABLE:
                st.info("Run: python -m spacy download en_core_web_sm")
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                st.info("Run: pip install sentence-transformers")
        
        st.markdown("---")
        st.header("💡 Smart Questions")
        
        smart_questions = [
            "What are the application deadlines?",
            "What qualifications do I need?",
            "How much does the program cost?",
            "What is the program format?",
            "Tell me about the curriculum"
        ]
        
        for q in smart_questions:
            if st.button(f"• {q}", key=f"ai_q_{hash(q)}"):
                st.session_state.ai_chat_history.append({"role": "user", "content": q})
                response = st.session_state.ai_response_generator.generate_ai_response(
                    q, st.session_state.ai_context
                )
                st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
                if CHROMA_AVAILABLE:
                    st.session_state.ai_chat_manager.add_message(
                        q, response, PRIMARY_LIST, 
                        st.session_state.ai_response_generator.nlp_processor.analyze_intent(q)
                    )
        
        if st.button("📊 Conversation Insights"):
            if CHROMA_AVAILABLE and st.session_state.ai_chat_manager.collection:
                similar_convos = st.session_state.ai_chat_manager.get_similar_conversations(
                    "admission process", limit=2
                )
                if similar_convos:
                    st.write("**Similar past conversations:**")
                    for convo in similar_convos:
                        st.caption(f"Q: {convo['query'][:50]}...")
                else:
                    st.info("No similar conversations found yet.")
            else:
                st.info("ChromaDB not available for conversation insights.")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 AI Chat")
        
        # Display enhanced chat history
        for message in st.session_state.ai_chat_history[-6:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show analysis if enabled
                if show_analysis and message["role"] == "user" and SPACY_AVAILABLE:
                    with st.expander("🔍 NLP Analysis"):
                        intent = st.session_state.ai_response_generator.nlp_processor.analyze_intent(
                            message["content"]
                        )
                        st.write(f"**Detected Intent:** {intent}")
                        entities = st.session_state.ai_response_generator.nlp_processor.extract_entities(
                            message["content"]
                        )
                        if entities:
                            st.write("**Entities Found:**")
                            for entity in entities[:3]:
                                st.write(f"- {entity['text']} ({entity['label']})")
                        else:
                            st.write("No named entities found.")
        
        # AI-enhanced chat input
        query = st.chat_input("Ask me anything about MDA Weekend...")
        
        if query:
            # Add user message
            st.session_state.ai_chat_history.append({"role": "user", "content": query})
            
            # Generate and display AI response
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Processing with AI..."):
                    # Find similar past conversations for context
                    similar_convos = []
                    if semantic_search and CHROMA_AVAILABLE:
                        similar_convos = st.session_state.ai_chat_manager.get_similar_conversations(query)
                    
                    # Generate response
                    if ai_enabled:
                        response = st.session_state.ai_response_generator.generate_ai_response(
                            query, st.session_state.ai_context
                        )
                    else:
                        response = st.session_state.ai_response_generator.generate_rule_based_response(
                            query, 
                            st.session_state.ai_response_generator.nlp_processor.analyze_intent(query),
                            ""
                        )
                    
                    st.write(response)
                    
                    # Show similar conversations if found
                    if similar_convos and show_analysis:
                        with st.expander("🔍 Similar Past Questions"):
                            for convo in similar_convos:
                                st.write(f"**Q:** {convo['query'][:60]}...")
                                st.write(f"**A:** {convo['response'][:80]}...")
                                st.markdown("---")
            
            # Add to history and AI database
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            if CHROMA_AVAILABLE:
                st.session_state.ai_chat_manager.add_message(
                    query, response, PRIMARY_LIST,
                    st.session_state.ai_response_generator.nlp_processor.analyze_intent(query)
                )
    
    with col2:
        st.subheader("🧠 AI Insights")
        
        # AI capabilities status
        st.write("**AI Capabilities:**")
        col1, col2 = st.columns(2)
        with col1:
            status = "✅" if SPACY_AVAILABLE else "❌"
            st.caption(f"NLP: {status}")
            st.caption("OpenAI: ❌ (Disabled)")
        with col2:
            status = "✅" if SENTENCE_TRANSFORMERS_AVAILABLE else "❌"
            st.caption(f"Semantic: {status}")
            status = "✅" if CHROMA_AVAILABLE else "❌"
            st.caption(f"ChromaDB: {status}")
        
        st.markdown("---")
        
        # Context information
        st.write("**Current Context:**")
        st.caption(f"Sentences: {len(st.session_state.ai_context)}")
        st.caption(f"Chat History: {len(st.session_state.ai_chat_history)} messages")
        
        # Quick analysis
        if st.session_state.ai_chat_history and SPACY_AVAILABLE:
            last_query = st.session_state.ai_chat_history[-1]["content"] 
            if st.session_state.ai_chat_history[-1]["role"] == "user":
                intent = st.session_state.ai_response_generator.nlp_processor.analyze_intent(last_query)
                st.write(f"**Last Intent:** {intent}")
        
        st.markdown("---")
        st.write("**💡 Tips**")
        st.caption("• Ask complex questions")
        st.caption("• Use natural language")
        st.caption("• Request comparisons")
        st.caption("• Ask about benefits")

if __name__ == "__main__":
    main()