import os
import uuid
import re
import requests
from datetime import datetime
from glob import glob
from dotenv import load_dotenv
import json
import shutil

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

import streamlit as st
st.set_page_config(
    page_title="MDA VCU 2025 Chatbot developed by MDA 2025 LUCKSTONE GROUP3 (Alejandro,Anisha,Jason,Noah,Som) : Demo on 14NOV2025 SAT", 
    layout="wide"
)

load_dotenv()

# Configuration - Use absolute path to ensure we're working with the right Data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FOLDER = os.path.join(BASE_DIR, "Data")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
CHROMA_HISTORY_PATH = os.path.join(BASE_DIR, "chroma_history")
OPENAI_API_KEY = "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"

# Default URLs for different categories
DEFAULT_URLS = {
    "url1": "https://business.vcu.edu/graduate-programs/mda-weekend/",
    "url2": "https://www.bbc.com/news",
    "url3": "https://www.wikipedia.org/",
    "url4": "https://www.nasa.gov/",
    "url5": "https://www.techcrunch.com/"
}

# --------- CLEANUP FUNCTIONS ---------
def clear_data_folder():
    """Clear all files from the Data folder for fresh start"""
    try:
        st.info(f"🔄 Clearing Data folder: {DOCUMENTS_FOLDER}")
        
        if os.path.exists(DOCUMENTS_FOLDER):
            # List files before deletion for debugging
            files_before = os.listdir(DOCUMENTS_FOLDER)
            st.info(f"📁 Files in Data folder before cleanup: {files_before}")
            
            # Remove all files in Data folder
            deleted_count = 0
            for filename in os.listdir(DOCUMENTS_FOLDER):
                file_path = os.path.join(DOCUMENTS_FOLDER, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        deleted_count += 1
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        deleted_count += 1
                except Exception as e:
                    st.warning(f"Could not delete {file_path}: {e}")
            
            # List files after deletion for debugging
            files_after = os.listdir(DOCUMENTS_FOLDER) if os.path.exists(DOCUMENTS_FOLDER) else []
            st.info(f"📁 Files in Data folder after cleanup: {files_after}")
            
            st.success(f"🗑️ Cleared {deleted_count} files from {DOCUMENTS_FOLDER}")
            return True
        else:
            # Create the folder if it doesn't exist
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            st.info(f"📁 Created Data folder: {DOCUMENTS_FOLDER}")
            return True
    except Exception as e:
        st.error(f"❌ Error clearing Data folder: {e}")
        return False

def perform_fresh_start():
    """Perform fresh start - clear only Data folder, preserve ChromaDB"""
    with st.spinner("🔄 Performing fresh start..."):
        # Clear Data folder only (preserve ChromaDB)
        data_cleared = clear_data_folder()
        
        # Clear session state (but preserve ChromaDB-related states)
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.phone_number = ""
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.qa_chain = None
        st.session_state.documents_loaded = False
        st.session_state.loaded_sources = []
        st.session_state.url_inputs = DEFAULT_URLS.copy()
        
        # Note: We intentionally DON'T clear doc_store and history_store
        # so historical data persists
        
        if data_cleared:
            st.success("✅ Fresh start completed! Data folder cleared but historical data preserved.")
        else:
            st.warning("⚠️ Some data may not have been cleared completely.")

# --------- URL DOWNLOAD FUNCTIONS ---------
def ensure_data_folder():
    """Ensure the Data folder exists and is properly structured"""
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    st.info(f"📁 Data folder location: {DOCUMENTS_FOLDER}")
    return DOCUMENTS_FOLDER

def download_url_content(url, filename, folder=DOCUMENTS_FOLDER):
    """Download webpage content and save as text file"""
    try:
        ensure_data_folder()
        
        # Create safe filename with proper numbering
        safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
        filepath = os.path.join(folder, f"{safe_filename}.txt")
        
        st.info(f"💾 Saving URL content to: {filepath}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Extract text content from HTML
            content = f"URL: {url}\nDownloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTitle: {filename}\n\n"
            
            # Try to extract meaningful text from HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get title if available
            title_tag = soup.find('title')
            if title_tag:
                content += f"Page Title: {title_tag.get_text()}\n\n"
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            content += f"Content extracted from {url}:\n\n{text[:20000]}"  # Increased content length
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify file was created
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                st.success(f"✅ Successfully saved: {os.path.basename(filepath)} ({file_size} bytes)")
                return filepath, file_size
            else:
                st.error(f"❌ File was not created: {filepath}")
                return None, 0
        else:
            st.error(f"❌ Failed to access {url}. Status code: {response.status_code}")
            # Still create a file with error information
            error_content = f"URL: {url}\nDownloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nError: Failed to access URL. Status code: {response.status_code}\n\nThis may require authentication or the URL may not be accessible."
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(error_content)
            return filepath, len(error_content)
            
    except Exception as e:
        st.error(f"❌ Error downloading {url}: {str(e)}")
        # Create error file for consistency
        safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
        error_filepath = os.path.join(folder, f"{safe_filename}_error.txt")
        with open(error_filepath, 'w', encoding='utf-8') as f:
            error_content = f"URL: {url}\nError: {str(e)}"
            f.write(error_content)
        return error_filepath, len(error_content)

def download_multiple_urls_sequential(url_dict):
    """Download multiple URLs sequentially one by one and ensure proper numbering"""
    downloaded_files = []
    
    # Ensure Data folder exists
    ensure_data_folder()
    
    # Filter out empty URLs and get only provided ones
    provided_urls = {k: v for k, v in url_dict.items() if v and v.strip()}
    
    if not provided_urls:
        st.warning("⚠️ No valid URLs provided")
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    with results_container:
        st.subheader("🌐 URL Download Progress")
    
    for i, (url_key, url) in enumerate(provided_urls.items()):
        status_text.text(f"📥 Downloading URL {i+1}/{len(provided_urls)}: {url[:80]}...")
        progress_bar.progress((i) / len(provided_urls))
        
        # Use consistent naming based on URL key (url1, url2, etc.)
        filename = f"url_content_{url_key[-1]}"  # Extract number from url1, url2, etc.
        filepath, file_size = download_url_content(url, filename)
        
        if filepath and os.path.exists(filepath):
            # Check if file has actual content (more lenient check)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content.strip()) > 50:  # Reduced minimum content check
                downloaded_files.append(filepath)
                with results_container:
                    st.success(f"✅ **URL {i+1}** ({url_key}): {url}")
                    st.info(f"   📁 Saved as: `{os.path.basename(filepath)}` ({file_size} bytes)")
            else:
                with results_container:
                    st.warning(f"⚠️ **URL {i+1}** ({url_key}): {url} - Minimal content")
        else:
            with results_container:
                st.error(f"❌ **URL {i+1}** ({url_key}): {url} - Download failed")
    
    progress_bar.progress(1.0)
    status_text.text("✅ URL download completed!")
    
    # Show summary of downloaded files
    if downloaded_files:
        with results_container:
            st.subheader("📋 Download Summary")
            for i, filepath in enumerate(downloaded_files):
                file_size = os.path.getsize(filepath)
                st.write(f"{i+1}. `{os.path.basename(filepath)}` - {file_size} bytes")
    
    return downloaded_files

def check_existing_url_files():
    """Check what URL content files already exist in Data folder"""
    ensure_data_folder()
    existing_files = []
    
    for i in range(1, 6):
        filename = f"url_content_{i}.txt"
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            # Try to read the URL from the file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    url = first_line.replace('URL: ', '') if first_line.startswith('URL: ') else 'Unknown URL'
            except:
                url = 'Unknown URL'
            existing_files.append((filename, file_size, url))
    
    return existing_files

def clear_existing_url_files():
    """Clear existing URL files before new download"""
    ensure_data_folder()
    cleared_count = 0
    
    for i in range(1, 6):
        filename = f"url_content_{i}.txt"
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                cleared_count += 1
                st.info(f"🗑️ Removed: {filename}")
            except Exception as e:
                st.warning(f"Could not remove {filename}: {e}")
    
    return cleared_count

# --------- HISTORICAL DATA QUERY FUNCTIONS ---------
def get_all_conversation_history(history_store, limit=1000):
    """Get all conversation history from ChromaDB"""
    try:
        if not history_store:
            return []
            
        results = history_store.get()
        
        all_history = []
        for i, content in enumerate(results["documents"]):
            metadata = results["metadatas"][i]
            
            # Only include Q&A pairs
            if metadata.get("type") == "qa_pair":
                all_history.append({
                    "timestamp": metadata.get("timestamp", "Unknown"),
                    "username": metadata.get("username", "Unknown"),
                    "phone_number": metadata.get("phone_number", "Unknown"),
                    "question": metadata.get("question", "Unknown"),
                    "answer": metadata.get("answer", "Unknown"),
                    "source_count": metadata.get("source_count", 0),
                    "session_id": metadata.get("session_id", "Unknown"),
                    "unique_id": metadata.get("unique_id", "Unknown")
                })
        
        # Sort by timestamp (newest first)
        all_history.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_history[:limit]
        
    except Exception as e:
        st.error(f"❌ Error retrieving all history: {str(e)}")
        return []

def get_all_users(history_store):
    """Get list of all unique users from historical data"""
    try:
        if not history_store:
            return []
            
        history = get_all_conversation_history(history_store)
        users = set()
        
        for entry in history:
            username = entry.get("username")
            phone = entry.get("phone_number")
            if username and phone:
                users.add((username, phone))
        
        return list(users)
    except Exception as e:
        st.error(f"❌ Error getting users: {str(e)}")
        return []

def get_user_statistics(history_store, username, phone_number):
    """Get statistics for a specific user"""
    try:
        if not history_store:
            return {}
            
        user_history = get_user_history_chroma(username, phone_number, history_store, limit=10000)
        
        if not user_history:
            return {}
        
        # Calculate statistics
        total_questions = len(user_history)
        
        # Count questions by date
        date_counts = {}
        for entry in user_history:
            date = entry["timestamp"][:10]  # Extract YYYY-MM-DD
            date_counts[date] = date_counts.get(date, 0) + 1
        
        # Find most active date
        most_active_date = max(date_counts, key=date_counts.get) if date_counts else "None"
        most_active_count = date_counts.get(most_active_date, 0)
        
        return {
            "total_questions": total_questions,
            "first_interaction": user_history[-1]["timestamp"] if user_history else "None",
            "last_interaction": user_history[0]["timestamp"] if user_history else "None",
            "most_active_date": most_active_date,
            "questions_on_most_active_date": most_active_count,
            "unique_days": len(date_counts)
        }
    except Exception as e:
        st.error(f"❌ Error getting user statistics: {str(e)}")
        return {}

# --------- PDF PROCESSING FIXES ---------
def extract_text_from_pdf_fallback(file_path):
    """Multiple fallback methods for PDF text extraction"""
    file_name = os.path.basename(file_path)
    extracted_text = ""
    
    # Method 1: Try PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                extracted_text += f"Page {page_num + 1}:\n{text}\n\n"
        doc.close()
        if extracted_text.strip():
            st.success(f"✅ PyMuPDF (fitz) successful for {file_name}")
            return extracted_text
    except Exception as e:
        st.warning(f"⚠️ PyMuPDF failed for {file_name}: {str(e)}")
    
    # Method 2: Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    extracted_text += f"Page {page_num + 1}:\n{text}\n\n"
        if extracted_text.strip():
            st.success(f"✅ pdfplumber successful for {file_name}")
            return extracted_text
    except Exception as e:
        st.warning(f"⚠️ pdfplumber failed for {file_name}: {str(e)}")
    
    # Method 3: Try PyPDF2
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text += f"Page {page_num + 1}:\n{text}\n\n"
        if extracted_text.strip():
            st.success(f"✅ PyPDF2 successful for {file_name}")
            return extracted_text
    except Exception as e:
        st.warning(f"⚠️ PyPDF2 failed for {file_name}: {str(e)}")
    
    # Method 4: Try OCR if all else fails (requires installation)
    try:
        st.info(f"🔄 Attempting OCR for {file_name}...")
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            st.warning("⚠️ OCR libraries not installed. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract", "pillow"])
            import pytesseract
            from PIL import Image
            
        # Convert PDF to images and OCR
        import fitz
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")
            
            # Convert to PIL Image and OCR
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img)
            if text.strip():
                extracted_text += f"Page {page_num + 1} (OCR):\n{text}\n\n"
        
        doc.close()
        if extracted_text.strip():
            st.success(f"✅ OCR successful for {file_name}")
            return extracted_text
    except Exception as e:
        st.warning(f"⚠️ OCR failed for {file_name}: {str(e)}")
    
    return None

def load_and_split_document_robust(file_path: str):
    """Robust document loading with multiple fallback methods"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        st.write(f"🔄 Processing: {file_name}")
        
        documents = []
        
        if file_ext == '.pdf':
            # Try multiple PDF extraction methods
            extracted_text = extract_text_from_pdf_fallback(file_path)
            
            if extracted_text and extracted_text.strip():
                document = Document(
                    page_content=extracted_text, 
                    metadata={
                        "source_file": file_name, 
                        "full_path": file_path,
                        "file_type": file_ext,
                        "method": "multiple_fallbacks"
                    }
                )
                documents = [document]
            else:
                st.error(f"❌ All PDF extraction methods failed for {file_name}")
                # Create a minimal document with file info
                document = Document(
                    page_content=f"Document: {file_name}\nFile Type: PDF\nStatus: Could not extract text content. This may be a scanned PDF or image-based document.",
                    metadata={
                        "source_file": file_name, 
                        "full_path": file_path,
                        "file_type": file_ext,
                        "method": "minimal_fallback"
                    }
                )
                documents = [document]
                st.warning(f"⚠️ Created minimal document for {file_name}")
                
        elif file_ext in ['.docx', '.doc']:
            try:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                st.success(f"✅ Successfully loaded DOCX: {file_name}")
            except Exception as e:
                st.error(f"❌ DOCX loader failed for {file_name}: {str(e)}")
                return []
                
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                document = Document(
                    page_content=content, 
                    metadata={
                        "source_file": file_name, 
                        "full_path": file_path,
                        "file_type": file_ext
                    }
                )
                documents = [document]
                st.success(f"✅ Successfully loaded TXT: {file_name}")
            except Exception as e:
                st.error(f"❌ TXT loader failed for {file_name}: {str(e)}")
                return []
                
        else:
            st.warning(f"⚠️ Unsupported file type: {file_ext}")
            return []
        
        # Process documents into chunks
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Filter out empty chunks
            non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            st.success(f"✅ Created {len(non_empty_chunks)} chunks from {file_name}")
            return non_empty_chunks
        else:
            st.warning(f"⚠️ No documents loaded from {file_name}")
            return []
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading {file_path}: {str(e)}")
        return []

# --------- LOGO + SUBHEADER HELPER ---------
def subheader_with_logo(text, logo_path="mda_logo.png", logo_width=36):
    try:
        cols = st.columns([1, 15])
        with cols[0]:
            if os.path.exists(logo_path):
                st.image(logo_path, width=logo_width)
        with cols[1]:
            st.subheader(text)
    except:
        st.subheader(text)

# --------- FORM FILE DROPDOWN + DOWNLOAD ---------
def get_form_files(folder):
    form_files = []
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "form" in file.lower():
                    form_files.append(os.path.join(root, file))
    return form_files

def extract_form_number(filename):
    m = re.search(r'form[_\s-]*(\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

# --------- CHROMADB FUNCTIONS ---------
def initialize_chroma_stores():
    """Initialize both document and history ChromaDB stores"""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Document store for policy documents
        doc_store = Chroma(
            collection_name="mda_documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # History store for conversations
        history_store = Chroma(
            collection_name="mda_history",
            embedding_function=embeddings,
            persist_directory=CHROMA_HISTORY_PATH
        )
        
        return doc_store, history_store, embeddings
    except Exception as e:
        st.error(f"❌ Error initializing ChromaDB stores: {str(e)}")
        return None, None, None

def save_conversation_to_chroma(username, phone_number, question, answer, source_docs, history_store):
    """Save conversation to ChromaDB history store"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        unique_id = str(uuid.uuid4())
        session_id = st.session_state.get('session_id', str(uuid.uuid4()))
        
        # Create main Q&A document
        qa_doc = Document(
            page_content=f"Question: {question}\n\nAnswer: {answer}",
            metadata={
                "type": "qa_pair",
                "username": username,
                "phone_number": phone_number,
                "question": question,
                "answer": answer,
                "timestamp": timestamp,
                "unique_id": unique_id,
                "session_id": session_id,
                "business_date": timestamp[:10],
                "source_count": len(source_docs)
            }
        )
        
        # Prepare documents list
        docs_to_add = [qa_doc]
        
        # Add source documents with metadata
        for i, doc in enumerate(source_docs):
            source_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "type": "source_reference",
                    "username": username,
                    "phone_number": phone_number,
                    "question": question,
                    "timestamp": timestamp,
                    "unique_id": unique_id,
                    "session_id": session_id,
                    "business_date": timestamp[:10],
                    "relevance_rank": i + 1,
                    "url_file_path": doc.metadata.get('full_path', ''),
                    "source_file": doc.metadata.get('source_file', '')
                }
            )
            docs_to_add.append(source_doc)
        
        # Add to ChromaDB
        history_store.add_documents(docs_to_add)
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving conversation: {str(e)}")
        return False

def check_user_exists_chroma(username, phone_number, history_store):
    """Check if user exists in ChromaDB history"""
    try:
        results = history_store.get()
        
        for metadata in results["metadatas"]:
            if (metadata.get("username") == username and 
                metadata.get("phone_number") == phone_number and
                metadata.get("type") == "qa_pair"):
                return True
        return False
        
    except Exception as e:
        st.error(f"❌ Error checking user existence: {str(e)}")
        return False

def get_user_history_chroma(username, phone_number, history_store, limit=50):
    """Get user's chat history from ChromaDB"""
    try:
        results = history_store.get()
        
        history = []
        for i, content in enumerate(results["documents"]):
            metadata = results["metadatas"][i]
            
            # Filter for this user's Q&A pairs
            if (metadata.get("username") == username and 
                metadata.get("phone_number") == phone_number and
                metadata.get("type") == "qa_pair"):
                
                history.append({
                    "timestamp": metadata.get("timestamp"),
                    "question": metadata.get("question"),
                    "answer": metadata.get("answer"),
                    "source_count": metadata.get("source_count", 0),
                    "session_id": metadata.get("session_id")
                })
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history[:limit]
        
    except Exception as e:
        st.error(f"❌ Error retrieving history: {str(e)}")
        return []

# --------- DOCUMENT PROCESSING ---------
def get_supported_documents(folder):
    """Get all supported documents (PDF, DOCX, TXT) from folder"""
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    documents = []
    
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    documents.append(os.path.join(root, file))
    
    return documents

def initialize_vector_store_with_all_sources(url_dict=None):
    """Initialize vector store with all documents from Data folder and URLs"""
    try:
        if not OPENAI_API_KEY:
            st.error("❌ OpenAI API key not found.")
            return None, None, None
            
        doc_store, history_store, embeddings = initialize_chroma_stores()
        if not doc_store or not history_store:
            return None, None, None
        
        all_chunks = []
        processed_files = []
        failed_files = []
        
        # Step 1: Check existing URL files
        st.subheader("📁 Checking Existing Documents")
        existing_url_files = check_existing_url_files()
        if existing_url_files:
            st.info("📋 Existing URL files found:")
            for filename, size, url in existing_url_files:
                st.write(f"  - {filename} ({size} bytes) - {url}")
        else:
            st.info("📁 No existing URL files found in Data folder")
        
        # Step 2: Process all existing documents from Data folder
        local_documents = get_supported_documents(DOCUMENTS_FOLDER)
        
        if local_documents:
            st.success(f"Found {len(local_documents)} supported documents in Data folder:")
            for doc in local_documents:
                st.write(f"  - {os.path.basename(doc)}")
            
            # Process files sequentially with progress
            progress_bar = st.progress(0)
            for i, file_path in enumerate(local_documents):
                progress_bar.progress(i / len(local_documents))
                chunks = load_and_split_document_robust(file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(os.path.basename(file_path))
                    st.success(f"✅ Successfully processed {os.path.basename(file_path)} - {len(chunks)} chunks")
                else:
                    failed_files.append(os.path.basename(file_path))
                    st.error(f"❌ Failed to extract content from {os.path.basename(file_path)}")
            
            progress_bar.progress(1.0)
        else:
            st.warning("⚠️ No supported documents found in Data folder")
        
        # Step 3: Process URLs if provided
        if url_dict and any(url.strip() for url in url_dict.values()):
            st.subheader("🌐 Processing URLs")
            
            # Filter out empty URLs
            provided_urls = {k: v for k, v in url_dict.items() if v and v.strip()}
            st.info(f"Loading {len(provided_urls)} provided URLs")
            
            # Clear existing URL files to avoid conflicts
            cleared_count = clear_existing_url_files()
            if cleared_count > 0:
                st.info(f"🔄 Cleared {cleared_count} existing URL files")
            
            # Download URLs sequentially
            url_files = download_multiple_urls_sequential(provided_urls)
            
            # Process downloaded URL files
            for url_file in url_files:
                chunks = load_and_split_document_robust(url_file)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(f"URL: {os.path.basename(url_file)}")
                    st.success(f"✅ Successfully processed URL content - {len(chunks)} chunks")
                else:
                    st.error(f"❌ Failed to extract content from URL file")
        
        # Show summary
        if failed_files:
            st.warning(f"⚠️ Failed to process {len(failed_files)} files: {', '.join(failed_files)}")
        
        # Check if we have any content
        if not all_chunks:
            st.error("❌ No content could be extracted from any sources.")
            return None, None, None
        
        # Clear existing documents and add new ones
        try:
            doc_store.delete_collection()
            st.info("🔄 Cleared existing document collection")
        except:
            st.info("🔄 Creating new document collection")
        
        doc_store = Chroma(
            collection_name="mda_documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # Add documents in batches with progress
        batch_size = 100
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        with st.spinner(f"Adding {len(all_chunks)} chunks to vector database..."):
            progress_bar = st.progress(0)
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                doc_store.add_documents(batch)
                progress_bar.progress(min((i + batch_size) / len(all_chunks), 1.0))
                st.write(f"  ✅ Added batch {i//batch_size + 1}/{total_batches}")
        
        st.success(f"✅ Created vector store with {len(all_chunks)} chunks from {len(processed_files)} sources")
        st.success("📊 Sources loaded: " + ", ".join(processed_files))
        return doc_store, history_store, embeddings
        
    except Exception as e:
        st.error(f"❌ Error initializing vector store: {str(e)}")
        return None, None, None

def create_qa_chain(vectorstore, embeddings):
    if not vectorstore:
        return None
        
    # Updated prompt to handle any topic from the loaded documents
    custom_prompt = PromptTemplate(
        template="""You are a helpful AI assistant that answers questions based on the provided context.

Context: {context}
Question: {question}

Instructions:
1. Answer questions based ONLY on the provided context from the loaded documents and URLs
2. If the information is not in the context, say "I don't have information about that in my knowledge base."
3. Be specific and cite relevant sources when possible
4. Format your response clearly with bullet points or sections when appropriate
5. If the question is unclear, ask for clarification

Answer:""",
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    
    return qa_chain

def find_document(file_name):
    if not os.path.exists(DOCUMENTS_FOLDER):
        return None
        
    for root, dirs, files in os.walk(DOCUMENTS_FOLDER):
        for file in files:
            if file_name.lower() in file.lower():
                return os.path.join(root, file)
    return None

# --------- MAIN APPLICATION ---------
def main():
    # Show the actual Data folder location
    st.sidebar.info(f"📍 **Data Folder Location:**\n`{DOCUMENTS_FOLDER}`")
    
    # Initialize ChromaDB stores on first run (but don't clear them)
    if "chroma_initialized" not in st.session_state:
        with st.spinner("🔄 Loading historical data..."):
            doc_store, history_store, embeddings = initialize_chroma_stores()
            if doc_store and history_store:
                st.session_state.doc_store = doc_store
                st.session_state.history_store = history_store
                st.session_state.chroma_initialized = True
                
                # Show historical data stats
                all_history = get_all_conversation_history(history_store)
                if all_history:
                    st.sidebar.info(f"📊 Loaded {len(all_history)} historical conversations")
    
    # Header with logo
    try:
        if os.path.exists("mda_logo.png"):
            st.image("mda_logo.png", width=180)
    except:
        pass
        
    st.markdown(
        """
        <h1 style='text-align: left; color: #f46e1e;'>Multi-Source Knowledge Chatbot</h1>
        <h5 style='text-align: left; color: #0000FF;'>Powered by ChromaDB, LangChain, and OpenAI</h4>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
    """
    <p style='color: maroon; font-size: 16px;'>
        🚀 <strong>Chatbot developed by VCU MDA Group3 2025 (Alejandro, Anisha, Jason, Noah, Som)
        </strong>
    </p>
    """, 
    unsafe_allow_html=True
    )

    st.markdown(
    """
    <p style='color: #006A4E; font-size: 16px; font-weight: bold;'>
        Demo on 15NOV2025 NLP & AI - Historical data preserved across sessions
    </p>
    """,
    unsafe_allow_html=True
    )

    # Session state initialization
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "phone_number" not in st.session_state:
        st.session_state.phone_number = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "loaded_sources" not in st.session_state:
        st.session_state.loaded_sources = []
    if "url_inputs" not in st.session_state:
        st.session_state.url_inputs = DEFAULT_URLS.copy()

    # ===== SIDEBAR =====
    with st.sidebar:
        # Document loading section
        st.markdown("---")
        st.subheader("📚 Document Configuration")
        
        # Show current Data folder contents
        st.write("**📁 Current Data Folder Contents:**")
        if os.path.exists(DOCUMENTS_FOLDER):
            current_files = os.listdir(DOCUMENTS_FOLDER)
            if current_files:
                for file in current_files:
                    file_path = os.path.join(DOCUMENTS_FOLDER, file)
                    file_size = os.path.getsize(file_path)
                    st.write(f"• {file} ({file_size} bytes)")
            else:
                st.info("📁 Data folder is empty")
        else:
            st.error("❌ Data folder does not exist")
        
        # Check existing URL files
        existing_files = check_existing_url_files()
        if existing_files:
            st.write("**📋 Existing URL files:**")
            for filename, size, url in existing_files:
                st.write(f"• {filename} ({size} bytes)")
                st.write(f"  ↳ {url}")
        
        # URL input section - 5 separate URL inputs
        st.write("**🌐 Enter URLs to Load (Max 5 URLs)**")
        
        # Create 5 URL input fields
        url_inputs = {}
        cols = st.columns(2)
        
        with cols[0]:
            url_inputs["url1"] = st.text_input(
                "URL 1 (e.g., University):",
                value=st.session_state.url_inputs.get("url1", ""),
                placeholder="https://business.vcu.edu/graduate-programs/mda-weekend/",
                key="url1_input"
            )
            
            url_inputs["url2"] = st.text_input(
                "URL 2 (e.g., News):",
                value=st.session_state.url_inputs.get("url2", ""),
                placeholder="https://www.bbc.com/news",
                key="url2_input"
            )
            
            url_inputs["url3"] = st.text_input(
                "URL 3 (e.g., Encyclopedia):",
                value=st.session_state.url_inputs.get("url3", ""),
                placeholder="https://www.wikipedia.org/",
                key="url3_input"
            )
        
        with cols[1]:
            url_inputs["url4"] = st.text_input(
                "URL 4 (e.g., Science):",
                value=st.session_state.url_inputs.get("url4", ""),
                placeholder="https://www.nasa.gov/",
                key="url4_input"
            )
            
            url_inputs["url5"] = st.text_input(
                "URL 5 (e.g., Technology):",
                value=st.session_state.url_inputs.get("url5", ""),
                placeholder="https://www.techcrunch.com/",
                key="url5_input"
            )
        
        # Update session state with current inputs
        st.session_state.url_inputs = url_inputs
        
        # Count provided URLs
        provided_urls = {k: v for k, v in url_inputs.items() if v and v.strip()}
        st.write(f"**📊 URLs Provided:** {len(provided_urls)}/5")
        
        if provided_urls:
            st.write("**URLs to be loaded:**")
            for url_key, url in provided_urls.items():
                st.write(f"• {url_key}: {url}")
        else:
            st.info("💡 Enter URLs in the fields above to load content")
        
        # Load documents button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load All Documents & URLs", key="load_all", type="primary"):
                if not any(url_inputs.values()):
                    st.error("❌ No URLs provided")
                else:
                    with st.spinner("Loading all documents and URLs..."):
                        # Initialize vector store with all sources
                        doc_store, history_store, embeddings = initialize_vector_store_with_all_sources(url_inputs)
                        if doc_store and history_store:
                            st.session_state.doc_store = doc_store
                            st.session_state.history_store = history_store
                            st.session_state.qa_chain = create_qa_chain(doc_store, embeddings)
                            st.session_state.documents_loaded = True
                            
                            # Track loaded sources
                            provided_urls_list = [f"URL {i+1}: {url}" for i, (url_key, url) in enumerate(provided_urls.items())]
                            st.session_state.loaded_sources = provided_urls_list
                            
                            st.success("✅ All documents and URLs loaded successfully!")
                        else:
                            st.error("❌ Failed to load documents and URLs")
        
        with col2:
            if st.button("🗑️ Clear Current Data", key="clear_data", type="secondary"):
                perform_fresh_start()
                st.rerun()

        if st.session_state.documents_loaded:
            st.success("📄 Documents are loaded and ready!")
            st.write("**Loaded Sources:**")
            for source in st.session_state.loaded_sources:
                st.write(f"• {source}")

        # Historical Data Section
        st.markdown("---")
        st.subheader("📊 Historical Data")
        
        if st.session_state.history_store:
            all_history = get_all_conversation_history(st.session_state.history_store, limit=50)
            if all_history:
                st.success(f"💾 {len(all_history)} historical conversations available")
                
                # Show user statistics
                all_users = get_all_users(st.session_state.history_store)
                if all_users:
                    st.write(f"**👥 Total Users:** {len(all_users)}")
                    
                    # Current user stats if authenticated
                    if st.session_state.authenticated:
                        user_stats = get_user_statistics(
                            st.session_state.history_store,
                            st.session_state.username,
                            st.session_state.phone_number
                        )
                        if user_stats:
                            st.write(f"**📈 Your Stats:**")
                            st.write(f"• Total Questions: {user_stats['total_questions']}")
                            st.write(f"• Active Days: {user_stats['unique_days']}")
                            st.write(f"• Most Active: {user_stats['most_active_date']} ({user_stats['questions_on_most_active_date']} questions)")
            else:
                st.info("No historical data yet. Start chatting to build history!")
        else:
            st.info("Historical data storage not available")

        # Installation instructions for PDF issues
        with st.expander("🔧 PDF Extraction Troubleshooting"):
            st.write("""
            If PDF extraction fails, install these dependencies:
            ```bash
            pip install pymupdf pdfplumber PyPDF2 pytesseract pillow
            ```
            
            For OCR (scanned PDFs):
            ```bash
            # On Windows, also install Tesseract:
            # Download from: https://github.com/UB-Mannheim/tesseract/wiki
            ```
            """)

        subheader_with_logo("🛠️ Features")
        
        # --- Forms dropdown/download ---
        subheader_with_logo("🗂️ Download Form", logo_path="mda_logo.png", logo_width=58)
        form_files = get_form_files(DOCUMENTS_FOLDER)
        form_numbers, form_map = [], {}
        
        for filepath in form_files:
            fn = os.path.basename(filepath)
            form_num = extract_form_number(fn)
            label = f"Form {form_num} ({fn})" if form_num else fn
            form_numbers.append(label)
            form_map[label] = filepath
            
        if form_numbers:
            selected = st.selectbox("Select a form to download", form_numbers, key="form_selectbox")
            if selected in form_map:
                try:
                    with open(form_map[selected], "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        "📥 Download selected form", 
                        file_bytes, 
                        file_name=os.path.basename(form_map[selected]), 
                        key="form_dl"
                    )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            st.info("No forms found.")

        # --- Document request by name ---
        subheader_with_logo("📄 Document Request", logo_path="mda_logo.png", logo_width=48)
        doc_name = st.text_input("Enter document name:", key="doc_name_input")
        
        if st.button("🔍 Find Document"):
            if doc_name:
                file_path = find_document(doc_name)
                if file_path:
                    st.success(f"📄 Found: {os.path.basename(file_path)}")
                    try:
                        with open(file_path, 'rb') as file:
                            st.download_button(
                                label="📥 Download",
                                data=file.read(),
                                file_name=os.path.basename(file_path),
                                mime="application/octet-stream",
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("❌ Document not found.")

        # --- Actions ---
        subheader_with_logo("⚙️ Actions", logo_path="mda_logo.png", logo_width=48)
        if st.button("🔄 Logout & Clear Data", type="primary"):
            perform_fresh_start()
            st.rerun()
            
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # ===== AUTHENTICATION =====
    if not st.session_state.authenticated:
        subheader_with_logo("🔐 User Authentication")
        with st.form("login_form"):
            username = st.text_input("👤 Enter your username:")
            phone_number = st.text_input("📱 Enter your phone number:")
            login_button = st.form_submit_button("🚀 Start Chat")
            
            if login_button:
                if username.strip() and phone_number.strip():
                    st.session_state.username = username.strip()
                    st.session_state.phone_number = phone_number.strip()
                    st.session_state.authenticated = True
                    
                    # Check if user exists in ChromaDB
                    if st.session_state.history_store and check_user_exists_chroma(username, phone_number, st.session_state.history_store):
                        user_stats = get_user_statistics(st.session_state.history_store, username, phone_number)
                        if user_stats:
                            st.success(f"👋 Welcome back, {username}! You've asked {user_stats['total_questions']} questions across {user_stats['unique_days']} days.")
                        else:
                            st.success(f"👋 Welcome back, {username}!")
                    else:
                        st.success(f"🆕 Welcome new user, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Please enter both username and phone number.")

    # ===== MAIN CHAT INTERFACE =====
    else:
        subheader_with_logo(f"👋 Welcome, {st.session_state.username}!")
        
        # Document status
        if st.session_state.documents_loaded:
            st.success("✅ Documents and URLs are loaded and ready!")
            if st.session_state.loaded_sources:
                st.write("**📊 Loaded Sources:**")
                for source in st.session_state.loaded_sources:
                    st.write(f"• {source}")
        else:
            st.warning("⚠️ Please load documents first using the sidebar")
        
        subheader_with_logo("💬 Chat Interface")
        
        # Check if QA chain is available
        if not st.session_state.qa_chain:
            st.error("❌ Chat functionality not available. Please load documents first.")
        else:
            with st.form("chat_form", clear_on_submit=True):
                user_question = st.text_input("Ask a question about the loaded content:")
                submit_button = st.form_submit_button("🚀 Send")
                
                if submit_button and user_question:
                    try:
                        with st.spinner("🔍 Searching through loaded documents..."):
                            result = st.session_state.qa_chain.invoke({"query": user_question})
                            answer = result["result"]
                            source_docs = result["source_documents"]
                            
                            # Save to ChromaDB
                            if st.session_state.history_store:
                                save_conversation_to_chroma(
                                    st.session_state.username,
                                    st.session_state.phone_number,
                                    user_question,
                                    answer,
                                    source_docs,
                                    st.session_state.history_store
                                )
                            
                            # Add to session state
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": answer,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sources": source_docs
                            })
                            
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")

        # Current session chat history
        subheader_with_logo("📝 Current Session Chat")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                    st.markdown(f"**🕒 {chat['timestamp']}**")
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")
                    if chat['sources']:
                        st.markdown("**📚 Sources:**")
                        for j, doc in enumerate(chat['sources'][:3]):
                            st.markdown(f"- {doc.metadata.get('source_file', 'Unknown')}")
                            st.markdown(f"  *{doc.page_content[:200]}...*")
        else:
            st.info("No chat history yet. Start by asking a question!")

        # Complete user history from ChromaDB
        if st.session_state.history_store:
            with st.expander("📊 Your Complete Chat History (All Sessions)"):
                history = get_user_history_chroma(st.session_state.username, st.session_state.phone_number, st.session_state.history_store, limit=100)
                if history:
                    st.write(f"**📈 Total Questions:** {len(history)}")
                    for entry in history:
                        st.markdown(f"**🕒 {entry['timestamp']}**")
                        st.markdown(f"**❓ Q:** {entry['question']}")
                        st.markdown(f"**🤖 A:** {entry['answer']}")
                        st.markdown(f"**📚 Sources used:** {entry['source_count']}")
                        st.markdown(f"**🔗 Session:** {entry['session_id'][:8]}...")
                        st.markdown("---")
                else:
                    st.info("No previous history found.")

        # All Historical Data (Admin View)
        if st.session_state.history_store and st.session_state.authenticated:
            with st.expander("🔍 All Historical Data (All Users)"):
                all_history = get_all_conversation_history(st.session_state.history_store, limit=100)
                if all_history:
                    st.write(f"**📊 Total Conversations in Database:** {len(all_history)}")
                    
                    # User statistics
                    all_users = get_all_users(st.session_state.history_store)
                    st.write(f"**👥 Unique Users:** {len(all_users)}")
                    
                    # Show recent conversations
                    st.write("**🕒 Recent Conversations:**")
                    for entry in all_history[:10]:  # Show last 10
                        st.markdown(f"**{entry['timestamp']}** - **{entry['username']}**")
                        st.markdown(f"**Q:** {entry['question'][:100]}...")
                        st.markdown(f"**A:** {entry['answer'][:100]}...")
                        st.markdown("---")
                else:
                    st.info("No historical data found in database.")

if __name__ == "__main__":
    main()