"""
AskPDF Backend — RAG-based PDF Q&A using LangGraph with SQLite persistence.

Components:
  - PDFProcessor   : Extract text → chunk → embed → FAISS index
  - MetadataStore  : SQLite store for thread & PDF metadata
  - RAG Graph      : LangGraph pipeline (retrieve → relevance check → generate / reject)
  - AskPDFChatbot  : Public facade consumed by the Streamlit frontend
"""

import os
import uuid
import sqlite3
from datetime import datetime
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import fitz  # PyMuPDF

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
FAISS_DIR = "askpdf_faiss_indexes"
METADATA_DB_PATH = "askpdf_metadata.db"
CHECKPOINT_DB_PATH = "askpdf_checkpoints.db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════════════════════
#  LangGraph State
# ══════════════════════════════════════════════════════════════════════════════
class AskPDFState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    context: str
    is_relevant: bool
    source_pages: list[int]


# ══════════════════════════════════════════════════════════════════════════════
#  Metadata Store  (threads + PDFs)
# ══════════════════════════════════════════════════════════════════════════════
class MetadataStore:
    """Lightweight SQLite store for thread and PDF metadata."""

    def __init__(self, db_path: str = METADATA_DB_PATH):
        self.db_path = db_path
        self._setup()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _setup(self):
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS pdfs "
                "(id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS threads "
                "(id TEXT PRIMARY KEY, title TEXT NOT NULL, pdf_id TEXT NOT NULL, "
                "created_at TEXT NOT NULL, FOREIGN KEY(pdf_id) REFERENCES pdfs(id))"
            )

    # ── PDFs ──────────────────────────────────────────────────────────────
    def save_pdf(self, pdf_id: str, filename: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO pdfs VALUES (?, ?, ?)",
                (pdf_id, filename, datetime.now().isoformat()),
            )

    # ── Threads ───────────────────────────────────────────────────────────
    def create_thread(self, thread_id: str, pdf_id: str, title: str = "New Chat"):
        with self._conn() as c:
            c.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?)",
                (thread_id, title, pdf_id, datetime.now().isoformat()),
            )

    def list_threads(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT t.id, t.title, t.created_at, p.filename "
                "FROM threads t JOIN pdfs p ON t.pdf_id = p.id "
                "ORDER BY t.created_at DESC"
            ).fetchall()
        return [
            {"id": r[0], "title": r[1], "created_at": r[2], "pdf_name": r[3]}
            for r in rows
        ]

    def get_thread(self, thread_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT t.id, t.title, t.pdf_id, p.filename "
                "FROM threads t JOIN pdfs p ON t.pdf_id = p.id WHERE t.id = ?",
                (thread_id,),
            ).fetchone()
        if r:
            return {"id": r[0], "title": r[1], "pdf_id": r[2], "pdf_name": r[3]}
        return None

    def update_title(self, thread_id: str, title: str):
        with self._conn() as c:
            c.execute("UPDATE threads SET title = ? WHERE id = ?", (title, thread_id))

    def delete_thread(self, thread_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM threads WHERE id = ?", (thread_id,))


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Processor  (extract → chunk → FAISS)
# ══════════════════════════════════════════════════════════════════════════════
class PDFProcessor:
    """Handles PDF ingestion and FAISS vector-store management."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        os.makedirs(FAISS_DIR, exist_ok=True)

    def ingest(self, pdf_bytes: bytes) -> str:
        """Extract text from *pdf_bytes*, build a FAISS index, and return a pdf_id."""
        pdf_id = str(uuid.uuid4())

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not any(page.get_text().strip() for page in doc):
            doc.close()
            raise ValueError("No extractable text found in the PDF.")

        # Split each page's text into chunks and track which page they came from
        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if not page_text.strip():
                continue
            chunks = self.splitter.split_text(page_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"page": page_num})
        doc.close()

        vs = FAISS.from_texts(all_chunks, self.embeddings, metadatas=all_metadatas)
        vs.save_local(os.path.join(FAISS_DIR, pdf_id))
        return pdf_id

    def load(self, pdf_id: str) -> FAISS:
        """Load a previously-saved FAISS index."""
        return FAISS.load_local(
            os.path.join(FAISS_DIR, pdf_id),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LangGraph RAG Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def _build_rag_graph(vector_store: FAISS, checkpointer):
    """Construct and compile the RAG graph."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # ── Nodes ─────────────────────────────────────────────────────────────
    def retrieve(state: AskPDFState) -> dict:
        docs = vector_store.similarity_search(state["query"], k=4)
        pages = sorted({d.metadata.get("page", 0) for d in docs if d.metadata.get("page")})
        return {
            "context": "\n\n---\n\n".join(d.page_content for d in docs),
            "source_pages": pages,
        }

    def check_relevance(state: AskPDFState) -> dict:
        prompt = (
            "You are a relevance judge. Given the document context and user query, "
            "decide whether the query can be meaningfully answered from the context.\n\n"
            f"Context:\n{state['context']}\n\n"
            f"Query: {state['query']}\n\n"
            "Reply with a single word: relevant or irrelevant."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"is_relevant": "relevant" in resp.content.strip().lower()}

    def generate(state: AskPDFState) -> dict:
        system = (
            "You are AskPDF — an AI assistant that answers questions using ONLY the "
            "provided document context. Be accurate, concise, and well-structured. "
            "Use markdown formatting (headings, bullets, bold) when helpful.\n\n"
            f"### Document Context\n{state['context']}"
        )
        history = list(state.get("messages") or [])
        resp = llm.invoke(
            [SystemMessage(content=system)]
            + history
            + [HumanMessage(content=state["query"])]
        )
        return {
            "messages": [
                HumanMessage(content=state["query"]),
                AIMessage(content=resp.content),
            ]
        }

    def reject(state: AskPDFState) -> dict:
        return {
            "messages": [
                HumanMessage(content=state["query"]),
                AIMessage(
                    content=(
                        "I am not able to answer this question as it is not related "
                        "to the uploaded document. Please ask something about the "
                        "PDF content."
                    )
                ),
            ]
        }

    # ── Wiring ────────────────────────────────────────────────────────────
    g = StateGraph(AskPDFState)
    g.add_node("retrieve", retrieve)
    g.add_node("check_relevance", check_relevance)
    g.add_node("generate", generate)
    g.add_node("reject", reject)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "check_relevance")
    g.add_conditional_edges(
        "check_relevance",
        lambda s: "generate" if s["is_relevant"] else "reject",
        {"generate": "generate", "reject": "reject"},
    )
    g.add_edge("generate", END)
    g.add_edge("reject", END)

    return g.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
#  Public Facade
# ══════════════════════════════════════════════════════════════════════════════
class AskPDFChatbot:
    """Single entry-point consumed by the frontend."""

    def __init__(self):
        self.meta = MetadataStore()
        self.pdf = PDFProcessor()
        # SQLite-backed LangGraph checkpointer for conversation memory
        self._ckpt_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._ckpt_conn)
        self._checkpointer.setup()
        self._graphs: dict[str, object] = {}  # pdf_id → compiled graph

    # ── helpers ────────────────────────────────────────────────────────────
    def _graph_for(self, pdf_id: str):
        """Return (and cache) a compiled RAG graph for the given pdf_id."""
        if pdf_id not in self._graphs:
            vs = self.pdf.load(pdf_id)
            self._graphs[pdf_id] = _build_rag_graph(vs, self._checkpointer)
        return self._graphs[pdf_id]

    # ── PDF operations ────────────────────────────────────────────────────
    def upload_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        """Process a PDF and return its pdf_id."""
        pdf_id = self.pdf.ingest(pdf_bytes)
        self.meta.save_pdf(pdf_id, filename)
        return pdf_id

    # ── Thread operations ─────────────────────────────────────────────────
    def new_thread(self, pdf_id: str, title: str = "New Chat") -> str:
        tid = str(uuid.uuid4())
        self.meta.create_thread(tid, pdf_id, title)
        return tid

    def list_threads(self) -> list[dict]:
        return self.meta.list_threads()

    def get_thread(self, thread_id: str) -> dict | None:
        return self.meta.get_thread(thread_id)

    def delete_thread(self, thread_id: str):
        self.meta.delete_thread(thread_id)

    def rename_thread(self, thread_id: str, title: str):
        self.meta.update_title(thread_id, title)

    # ── Chat ──────────────────────────────────────────────────────────────
    def ask(self, query: str, thread_id: str, pdf_id: str) -> tuple[str, list[int]]:
        """Send a query through the RAG graph and return (answer, source_pages)."""
        graph = self._graph_for(pdf_id)
        result = graph.invoke(
            {"query": query, "messages": [], "context": "", "is_relevant": False, "source_pages": []},
            {"configurable": {"thread_id": thread_id}},
        )
        pages = result.get("source_pages", [])
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage):
                return m.content, pages
        return "Sorry, I couldn't generate a response. Please try again.", pages

    def get_history(self, thread_id: str, pdf_id: str) -> list:
        """Load conversation messages from the LangGraph checkpoint."""
        graph = self._graph_for(pdf_id)
        try:
            snap = graph.get_state({"configurable": {"thread_id": thread_id}})
            return snap.values.get("messages", [])
        except Exception:
            return []
