"""
AskPDF Frontend — Modern Streamlit Chat Interface.

Run:  streamlit run askpdf_frontend.py
"""

import html as html_lib
import time

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from ask_pdf_rag_chatbot import AskPDFChatbot

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskPDF — AI Document Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (modern, Tailwind-inspired) ───────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #667eea;
    --primary-dark: #5a67d8;
    --accent: #764ba2;
    --bg-dark: #1a1a2e;
    --bg-darker: #16213e;
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Header Banner ─────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    padding: 1.6rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
}
.app-header h1 {
    color: #fff; font-size: 2.4rem; font-weight: 700;
    margin: 0; letter-spacing: -0.5px;
}
.app-header p {
    color: rgba(255,255,255,0.85); font-size: 0.95rem;
    margin: 0.35rem 0 0; font-weight: 300;
}

/* ── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, var(--bg-dark), var(--bg-darker)) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
section[data-testid="stSidebar"] label {
    color: #d1d5db !important;
}

.sidebar-brand {
    text-align: center;
    padding: 1.2rem 0 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.sidebar-brand h2 {
    font-size: 1.7rem; font-weight: 700; margin: 0;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-brand p {
    color: rgba(255,255,255,0.45); font-size: 0.78rem; margin: 0.3rem 0 0;
}

.pdf-badge {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #fff; padding: 0.35rem 0.9rem; border-radius: 20px;
    font-size: 0.82rem; display: inline-block; margin: 0.5rem 0 0.4rem;
    font-weight: 500; box-shadow: 0 2px 8px rgba(102,126,234,0.25);
}

/* ── Welcome Card ──────────────────────────────────────── */
.welcome-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin: 3rem auto;
    max-width: 560px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.welcome-card h3 {
    color: #1a1a2e; font-size: 1.35rem; font-weight: 700;
    margin-bottom: 0.75rem;
}
.welcome-card p { color: #6b7280; line-height: 1.7; }

/* ── Feature pills on welcome card ─────────────────────── */
.features {
    display: flex; gap: 0.6rem; justify-content: center;
    flex-wrap: wrap; margin-top: 1.5rem;
}
.feature-pill {
    background: #f3f4f6; border: 1px solid #e5e7eb;
    border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.82rem; color: #4b5563; font-weight: 500;
}

/* ── Buttons ───────────────────────────────────────────── */
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s ease !important;
}

/* ── Chat messages ─────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Divider ───────────────────────────────────────────── */
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ── Source page badges ────────────────────────────────── */
.source-pages {
    display: flex; gap: 0.4rem; flex-wrap: wrap;
    align-items: center; margin-top: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e5e7eb;
}
.source-label {
    font-size: 0.78rem; color: #6b7280; font-weight: 500;
    margin-right: 0.2rem;
}
.source-pill {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff; padding: 0.2rem 0.65rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
    box-shadow: 0 1px 4px rgba(102,126,234,0.25);
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Chatbot singleton (cached across Streamlit reruns) ───────────────────────
@st.cache_resource
def _get_chatbot(_version="v2"):
    return AskPDFChatbot()


bot = _get_chatbot()

# ─── Session-state defaults ──────────────────────────────────────────────────
for key, default in {
    "thread_id": None,
    "pdf_id": None,
    "pdf_name": None,
    "messages": [],
    "upload_key": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(
        '<div class="sidebar-brand">'
        "<h2>📄 AskPDF</h2>"
        "<p>AI-Powered Document Chat</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── PDF Upload ────────────────────────────────────────────────────────
    st.markdown("#### 📎 Upload PDF")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded:
        file_key = f"{uploaded.name}|{uploaded.size}"
        if st.session_state.upload_key != file_key:
            with st.spinner("⏳ Processing your PDF…"):
                try:
                    pdf_id = bot.upload_pdf(uploaded.read(), uploaded.name)
                    st.session_state.pdf_id = pdf_id
                    st.session_state.pdf_name = uploaded.name
                    st.session_state.upload_key = file_key
                    tid = bot.new_thread(pdf_id, f"Chat — {uploaded.name}")
                    st.session_state.thread_id = tid
                    st.session_state.messages = []
                except Exception as exc:
                    st.error(f"Failed to process PDF: {exc}")

    if st.session_state.pdf_name:
        safe_name = html_lib.escape(st.session_state.pdf_name)
        st.markdown(
            f'<span class="pdf-badge">📄 {safe_name}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── New Chat ──────────────────────────────────────────────────────────
    if st.session_state.pdf_id:
        if st.button("➕  New Chat", use_container_width=True, type="primary"):
            tid = bot.new_thread(
                st.session_state.pdf_id,
                f"Chat — {st.session_state.pdf_name}",
            )
            st.session_state.thread_id = tid
            st.session_state.messages = []
            st.rerun()

    # ── Chat History ──────────────────────────────────────────────────────
    st.markdown("#### 💬 Chat History")
    threads = bot.list_threads()

    if not threads:
        st.caption("No conversations yet.")
    else:
        for t in threads:
            col_label, col_del = st.columns([5, 1])
            is_active = st.session_state.thread_id == t["id"]

            with col_label:
                label = ("🔵 " if is_active else "") + t["title"]
                if st.button(
                    label, key=f"t_{t['id']}", use_container_width=True
                ):
                    info = bot.get_thread(t["id"])
                    if info:
                        st.session_state.thread_id = t["id"]
                        st.session_state.pdf_id = info["pdf_id"]
                        st.session_state.pdf_name = info["pdf_name"]
                        history = bot.get_history(t["id"], info["pdf_id"])
                        st.session_state.messages = [
                            {
                                "role": (
                                    "user"
                                    if isinstance(m, HumanMessage)
                                    else "assistant"
                                ),
                                "content": m.content,
                            }
                            for m in history
                        ]
                        st.rerun()

            with col_del:
                if st.button("🗑️", key=f"d_{t['id']}"):
                    bot.delete_thread(t["id"])
                    if st.session_state.thread_id == t["id"]:
                        st.session_state.thread_id = None
                        st.session_state.messages = []
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    '<div class="app-header">'
    "<h1>📄 AskPDF</h1>"
    "<p>Upload a document and ask anything about its content</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Empty / welcome state ────────────────────────────────────────────────────
if not st.session_state.pdf_id or not st.session_state.thread_id:
    st.markdown(
        '<div class="welcome-card">'
        "<h3>👋 Welcome to AskPDF</h3>"
        "<p>Upload a PDF document using the sidebar to get started.<br>"
        "Once uploaded, you can ask any question about its content and "
        "receive intelligent, context-aware answers powered by AI.</p>"
        '<div class="features">'
        '<span class="feature-pill">📄 PDF Upload</span>'
        '<span class="feature-pill">🧠 RAG Pipeline</span>'
        '<span class="feature-pill">💬 Multi-thread Chats</span>'
        '<span class="feature-pill">🗂️ Chat History</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

# ── Active chat ──────────────────────────────────────────────────────────────
else:
    # Render saved messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ── Streaming helper ─────────────────────────────────────────────────────────
def _stream_text(text: str):
    """Generator that yields words with a small delay for a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


# ── Chat input (always visible) ──────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your PDF…"):
    # Guard: no PDF uploaded yet
    if not st.session_state.pdf_id or not st.session_state.thread_id:
        st.warning("📎 Please upload a PDF from the sidebar before asking questions.")
    else:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response with streaming effect
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, source_pages = bot.ask(
                        prompt,
                        st.session_state.thread_id,
                        st.session_state.pdf_id,
                    )
                except Exception as exc:
                    answer = f"⚠️ An error occurred: {exc}"
                    source_pages = []
            st.write_stream(_stream_text(answer))
            # Show source page references
            if source_pages:
                pills = " ".join(
                    f'<span class="source-pill">Page {p}</span>' for p in source_pages
                )
                st.markdown(
                    f'<div class="source-pages">'
                    f'<span class="source-label">📖 Sources:</span> {pills}'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        # Build display content for session history
        display_content = answer
        if source_pages:
            pages_text = ", ".join(f"Page {p}" for p in source_pages)
            display_content += f"\n\n📖 **Sources:** {pages_text}"
        st.session_state.messages.append(
            {"role": "assistant", "content": display_content}
        )

        # Auto-title the thread after the first exchange
        if len(st.session_state.messages) == 2:
            title = prompt[:50] + ("…" if len(prompt) > 50 else "")
            bot.rename_thread(st.session_state.thread_id, title)
