# 📄 AskPDF — AI-Powered PDF Chat

AskPDF is a **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions about their content. It uses OpenAI models for embeddings and answer generation, FAISS for vector similarity search, LangGraph for orchestrating the RAG pipeline, and Streamlit for the web interface.

---

## Features

- **PDF Upload & Text Extraction** — Upload any text-based PDF; pages are extracted using PyMuPDF.
- **RAG Pipeline** — Documents are chunked, embedded, and stored in a FAISS vector index. Queries retrieve the most relevant chunks, pass through a relevance check, and generate grounded answers.
- **Multi-Thread Conversations** — Create multiple chat threads per PDF, each with its own conversation history.
- **Conversation Persistence** — Thread metadata is stored in SQLite; conversation state is checkpointed via LangGraph's SQLite saver, so chats survive app restarts.
- **Source Page References** — Each answer shows which PDF pages the information came from.
- **Streaming-Style Responses** — Answers are displayed with a typing effect in the UI.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Streamlit Frontend  (askpdf_frontend.py)            │
│  ├─ PDF upload sidebar                               │
│  ├─ Chat history sidebar                             │
│  └─ Chat interface with streaming display            │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  AskPDFChatbot Facade  (ask_pdf_rag_chatbot.py)     │
│  ├─ PDFProcessor   → PyMuPDF + FAISS                │
│  ├─ MetadataStore  → SQLite (threads & PDFs)         │
│  └─ RAG Graph      → LangGraph pipeline              │
│       ┌─────────┐   ┌─────────────────┐             │
│       │Retrieve │──▶│Check Relevance  │             │
│       └─────────┘   └───────┬─────────┘             │
│                       relevant?                      │
│                     yes/        \no                   │
│               ┌──────────┐  ┌────────┐               │
│               │ Generate │  │ Reject │               │
│               └──────────┘  └────────┘               │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| LLM                | OpenAI `gpt-4o-mini`                |
| Embeddings         | OpenAI `text-embedding-3-small`     |
| Vector Store       | FAISS (local, file-persisted)       |
| Orchestration      | LangGraph (with SQLite checkpointer)|
| PDF Parsing        | PyMuPDF (`fitz`)                    |
| Metadata Storage   | SQLite                              |
| Frontend           | Streamlit                           |

---

## Prerequisites

- **Python 3.10+**
- An **OpenAI API key** with access to `gpt-4o-mini` and `text-embedding-3-small`

---

## Installation

### 1. Clone or download the project

```bash
cd "AskPDF Practice Chatbot"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-openai langchain-community langgraph faiss-cpu pymupdf python-dotenv
```

> **Note:** Use `faiss-cpu` for most setups. If you have a CUDA-capable GPU and want faster indexing on very large documents, you can install `faiss-gpu` instead.

### 4. Set up your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key-here
```

Replace `your-openai-api-key-here` with your actual key from [platform.openai.com](https://platform.openai.com/api-keys).

---

## Running the Application

```bash
streamlit run askpdf_frontend.py
```

Streamlit will start a local web server and open the app in your default browser (typically at `http://localhost:8501`).

---

## Usage

1. **Upload a PDF** — Use the sidebar file uploader to select a PDF document. The app extracts text, chunks it, generates embeddings, and builds a FAISS index. A new chat thread is created automatically.
2. **Ask questions** — Type a question in the chat input at the bottom. The RAG pipeline retrieves relevant chunks, checks if the query relates to the document, and generates an answer (or rejects irrelevant questions).
3. **View sources** — Each answer displays the PDF page numbers the information was drawn from.
4. **Manage conversations** — Create new chat threads with the **➕ New Chat** button, switch between threads in the sidebar, or delete old ones with the 🗑️ button.
5. **Upload another PDF** — Upload a different PDF at any time. Previous chat threads remain accessible in the sidebar.

---

## Project Structure

```
AskPDF Practice Chatbot/
├── ask_pdf_rag_chatbot.py   # Backend: RAG pipeline, PDF processing, metadata store
├── askpdf_frontend.py       # Frontend: Streamlit chat UI
├── .env                     # OpenAI API key (you create this)
├── askpdf_faiss_indexes/    # Auto-created: FAISS vector indexes per PDF
├── askpdf_metadata.db       # Auto-created: SQLite DB for thread/PDF metadata
├── askpdf_checkpoints.db    # Auto-created: SQLite DB for LangGraph conversation state
└── README.md
```

---

## Configuration

The following constants can be adjusted at the top of `ask_pdf_rag_chatbot.py`:

| Constant             | Default                    | Description                              |
|----------------------|----------------------------|------------------------------------------|
| `FAISS_DIR`          | `askpdf_faiss_indexes`     | Directory for persisted FAISS indexes    |
| `METADATA_DB_PATH`   | `askpdf_metadata.db`       | SQLite database for thread/PDF metadata  |
| `CHECKPOINT_DB_PATH` | `askpdf_checkpoints.db`    | SQLite database for LangGraph checkpoints|
| `EMBEDDING_MODEL`    | `text-embedding-3-small`   | OpenAI embedding model                   |
| `LLM_MODEL`          | `gpt-4o-mini`              | OpenAI chat model                        |

Chunk size and overlap for text splitting can be changed in the `PDFProcessor.__init__` method (defaults: 1000 chars / 200 overlap).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"No extractable text found in the PDF"** | The PDF is image-based (scanned). AskPDF requires text-based PDFs. Use an OCR tool first to convert it. |
| **OpenAI API errors** | Verify your API key in `.env` and ensure you have billing enabled on your OpenAI account. |
| **`ModuleNotFoundError`** | Make sure all dependencies are installed (`pip install ...` command above) and your virtual environment is activated. |
| **Port 8501 already in use** | Run with a different port: `streamlit run askpdf_frontend.py --server.port 8502` |

---

## License

This project is for educational / practice purposes.
