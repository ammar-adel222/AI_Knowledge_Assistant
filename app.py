import os
import sys
import io

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import shutil
import tempfile
import streamlit as st
from typing import List

from langchain_core.documents import Document
from model_configration import Config
from data_reader import data_extractor
from youtube_data_extractor import YouTubeExtractor
from youtube_summarizer import YouTubeSummarizer
from Rag__pipline import Chunker, Embedder, VectorStore
from chat import ChatEngine

# ══════════════════════════════════════════════════════════
# 1. STREAMLIT PAGE CONFIG & CUSTOM STYLING
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism aesthetic
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header gradient */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }

    /* Staged cards styling */
    .staged-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators */
    .status-ok {
        color: #22c55e;
        font-weight: 600;
    }
    .status-warn {
        color: #f59e0b;
        font-weight: 600;
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-pdf { background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .badge-sheet { background-color: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid #22c55e; }
    .badge-yt { background-color: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid #a855f7; }
</style>
""", unsafe_allow_html=True)

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 2. SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    # Try loading automatically if exists
    if st.session_state.vector_store.is_saved():
        try:
            st.session_state.vector_store.load(st.session_state.embedder)
        except Exception:
            pass

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine(st.session_state.vector_store)

if "staged_documents" not in st.session_state:
    st.session_state.staged_documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ══════════════════════════════════════════════════════════
# 3. SIDEBAR CONTROLS & DATA INGESTION
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 Knowledge Hub")
    
    # --- API Status & Input ---
    current_key = os.getenv("GROQ_API_KEY", Config.GROQ_API_KEY)
    user_api_key = st.text_input(
        "Groq API Key",
        value=current_key,
        type="password",
        help="Get a free Groq API key from https://console.groq.com/keys"
    )
    if user_api_key != current_key:
        os.environ["GROQ_API_KEY"] = user_api_key
        Config.GROQ_API_KEY = user_api_key
        # Update .env file
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"GROQ_API_KEY={user_api_key}\n")
        except Exception:
            pass
        st.success("API key updated!")
        st.rerun()

    if user_api_key:
        st.markdown("🟢 **Groq API**: Configured")
    else:
        st.markdown("🔴 **Groq API**: `GROQ_API_KEY` missing!")

    st.divider()

    # --- Knowledge Base Status & Load ---
    st.subheader("📦 Knowledge Base")
    is_saved = st.session_state.vector_store.is_saved()
    is_loaded = st.session_state.vector_store.store is not None

    if is_loaded:
        st.markdown("<span class='status-ok'>✅ Active & Ready</span>", unsafe_allow_html=True)
    elif is_saved:
        st.markdown("<span class='status-warn'>📂 Store Saved on Disk</span>", unsafe_allow_html=True)
        if st.button("Load Saved Knowledge Base", use_container_width=True):
            with st.spinner("Loading index from disk..."):
                st.session_state.vector_store.load(st.session_state.embedder)
                st.session_state.chat_engine = ChatEngine(st.session_state.vector_store)
                st.success("Loaded vector store successfully!")
                st.rerun()
    else:
        st.markdown("<span class='status-warn'>📭 No Store Built Yet</span>", unsafe_allow_html=True)

    st.divider()

    # --- Add New Data Sources ---
    st.subheader("📥 Add Data Sources")
    
    source_tab1, source_tab2, source_tab3 = st.tabs(["📄 PDF", "📊 CSV / Excel", "🎥 YouTube"])

    # --- PDF Tab ---
    with source_tab1:
        pdf_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
        if pdf_files and st.button("Add PDF Documents", key="btn_add_pdf"):
            extractor = data_extractor()
            added_count = 0
            with st.spinner("Processing PDF files..."):
                for pdf_file in pdf_files:
                    temp_path = os.path.join(TEMP_DIR, pdf_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    try:
                        docs = extractor.extract_pdf(temp_path)
                        st.session_state.staged_documents.extend(docs)
                        added_count += len(docs)
                    except Exception as e:
                        st.error(f"Error processing {pdf_file.name}: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            st.success(f"Added {added_count} pages to staging area!")

    # --- Excel / CSV Tab ---
    with source_tab2:
        sheet_files = st.file_uploader("Upload Spreadsheets", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="sheet_uploader")
        if sheet_files and st.button("Add Spreadsheets", key="btn_add_sheet"):
            extractor = data_extractor()
            added_count = 0
            with st.spinner("Processing spreadsheet files..."):
                for sheet_file in sheet_files:
                    temp_path = os.path.join(TEMP_DIR, sheet_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(sheet_file.getbuffer())
                    try:
                        docs = extractor.extract_spreadsheet(temp_path)
                        st.session_state.staged_documents.extend(docs)
                        added_count += len(docs)
                    except Exception as e:
                        st.error(f"Error processing {sheet_file.name}: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            st.success(f"Added {added_count} rows to staging area!")

    # --- YouTube Tab ---
    with source_tab3:
        yt_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        if yt_url and st.button("Extract & Summarize Video", key="btn_add_yt"):
            yt_extractor = YouTubeExtractor()
            yt_summarizer = YouTubeSummarizer()
            with st.spinner("Fetching transcript & summarizing with LLaMA..."):
                try:
                    docs = yt_extractor.extract(yt_url)
                    summarized_doc = yt_summarizer.summarize(docs[0])
                    st.session_state.staged_documents.append(summarized_doc)
                    st.success("Added YouTube summary to staging area!")
                    st.info(f"Summary: {summarized_doc.metadata['summary_word_count']} words ({summarized_doc.metadata['compression_percent']}% compression)")
                except Exception as e:
                    err_str = str(e)
                    if "401" in err_str or "invalid_api_key" in err_str or "Invalid API Key" in err_str:
                        st.error("🔑 **Invalid Groq API Key!** The key in your `.env` or sidebar is invalid or expired. Please enter a valid key in the sidebar (get one free at https://console.groq.com/keys).")
                    else:
                        st.error(f"Failed to process YouTube video: {e}")

    st.divider()

    # --- Staged Queue Status & Build Button ---
    st.subheader("📚 Staging Queue")
    staged_docs: List[Document] = st.session_state.staged_documents

    if staged_docs:
        counts = {}
        for d in staged_docs:
            stype = d.metadata.get("source_type", "unknown")
            counts[stype] = counts.get(stype, 0) + 1

        st.markdown(f"**Total Staged Items:** `{len(staged_docs)}`")
        for stype, count in counts.items():
            icon = {"pdf": "📄", "spreadsheet": "📊", "youtube": "🎥"}.get(stype, "📁")
            st.caption(f"{icon} **{stype.upper()}**: {count}")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🚀 Build Base", use_container_width=True):
                with st.spinner("Chunking & generating FAISS embeddings..."):
                    chunker = Chunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                    chunks = chunker.split(staged_docs)
                    st.session_state.vector_store.build(chunks, st.session_state.embedder)
                    st.session_state.vector_store.save()
                    st.session_state.chat_engine = ChatEngine(st.session_state.vector_store)
                    st.session_state.staged_documents = []
                    st.success("Knowledge Base Built & Saved!")
                    st.rerun()

        with col_b2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.staged_documents = []
                st.rerun()
    else:
        st.caption("No new items in staging queue.")


# ══════════════════════════════════════════════════════════
# 4. MAIN CHAT APPLICATION INTERFACE
# ══════════════════════════════════════════════════════════
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown('<div class="main-title">🤖 AI Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Source RAG Assistant powered by Groq LLaMA & FAISS</div>', unsafe_allow_html=True)

with header_col2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_engine.cache.clear()
        st.rerun()

st.divider()

# --- Render Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Cited Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# --- User Input ---
if prompt := st.chat_input("Ask a question about your documents or videos..."):
    if st.session_state.vector_store.store is None:
        st.warning("⚠️ Knowledge Base is empty! Please load or build a knowledge base first using the sidebar.")
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
                try:
                    # Search chunks for citation display
                    retrieved_chunks = st.session_state.chat_engine.qa_chain.search(prompt)
                    
                    # Get answer from chat engine
                    answer = st.session_state.chat_engine.chat(prompt)

                    st.markdown(answer)
                except Exception as e:
                    err_str = str(e)
                    if "401" in err_str or "invalid_api_key" in err_str or "Invalid API Key" in err_str:
                        answer = "🔑 **Invalid Groq API Key!** Please enter a valid API key in the sidebar to ask questions."
                        st.error(answer)
                        retrieved_chunks = []
                    else:
                        answer = f"Error generating answer: {e}"
                        st.error(answer)
                        retrieved_chunks = []

                # Format source citations
                sources = []
                for chunk in retrieved_chunks:
                    stype = chunk.metadata.get("source_type", "unknown")
                    if stype == "pdf":
                        info = f"📄 **PDF**: `{chunk.metadata.get('file_name', '?')}` (Page {chunk.metadata.get('page', '?')})"
                    elif stype == "spreadsheet":
                        info = f"📊 **Spreadsheet**: `{chunk.metadata.get('file_name', '?')}` (Row {chunk.metadata.get('row_number', '?')})"
                    elif stype == "youtube":
                        info = f"🎥 **YouTube**: Video ID `{chunk.metadata.get('video_id', '?')}`"
                    else:
                        info = f"📁 **Source**: {stype}"
                    if info not in sources:
                        sources.append(info)

                if sources:
                    with st.expander("📚 Cited Sources"):
                        for src in sources:
                            st.markdown(f"- {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
