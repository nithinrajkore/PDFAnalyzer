# PDF DataAnalyzer — RAG-Powered Document Q&A with Google Gemini

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.x-1C3C3C)](https://python.langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.0--flash-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent **PDF analysis and Q&A Streamlit application** that lets users upload multiple PDF documents and ask natural-language questions, powered by Google Gemini LLM, LangChain, and FAISS vector search (Retrieval-Augmented Generation).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Application Overview](#application-overview)
3. [RAG Architecture](#rag-architecture)
4. [Core Components](#core-components)
   - [PDF Text Extraction](#1-pdf-text-extraction)
   - [Text Chunking](#2-text-chunking)
   - [Vector Embeddings](#3-vector-embeddings)
   - [FAISS Vector Store](#4-faiss-vector-store)
   - [Conversational Q&A Chain](#5-conversational-qa-chain)
5. [Streamlit UI](#streamlit-ui)
6. [AWS S3 Integration](#aws-s3-integration)
7. [Environment Configuration](#environment-configuration)
8. [Tech Stack](#tech-stack)
9. [Getting Started](#getting-started)
10. [Key Concepts Glossary](#key-concepts-glossary)
11. [References](#references)

---

## Repository Structure

```
PDF-DataAnalyzer/
├── README.md
├── LICENSE                    ← MIT License
├── app.py                     ← Main Streamlit application + RAG pipeline
├── chat_ui.py                 ← Custom CSS for chat UI (user/bot message templates)
├── requirements.txt           ← Python dependencies
├── .gitignore
├── .env                       ← (local only) API keys — not committed
├── .streamlit/                ← Streamlit app configuration
├── assets/                    ← Static assets (images, icons)
└── .devcontainer/             ← Dev container config for VS Code / GitHub Codespaces
```

| File | Description |
|------|-------------|
| `app.py` | Core logic: PDF parsing, chunking, embedding, FAISS, Gemini Q&A chain |
| `chat_ui.py` | HTML/CSS templates for styled user and bot chat bubbles |
| `requirements.txt` | All dependencies: streamlit, langchain, faiss-cpu, PyPDF2, boto3, Google AI |
| `.streamlit/` | `config.toml` — theme, server settings |

---

## Application Overview

Users upload one or more PDF files → the app extracts text → splits into overlapping chunks → embeds via Google Gemini → stores in FAISS → user asks a question → relevant chunks are retrieved → Gemini generates a grounded answer → conversation history maintained.

**Key capabilities:**
- Multi-PDF upload and simultaneous analysis
- Conversational memory (tracks full chat history)
- Source document attribution (shows which chunks informed the answer)
- AWS S3 integration (PDF storage in cloud)
- Custom styled chat UI

---

## RAG Architecture

**Retrieval-Augmented Generation (RAG)** prevents LLM hallucination by grounding answers in retrieved document content:

```
PDF Files
    ↓
[1] Text Extraction (PyPDF2)
    ↓
[2] Text Chunking (CharacterTextSplitter, chunk_size=1000, overlap=200)
    ↓
[3] Embedding Generation (Google Gemini: models/embedding-001)
    ↓
[4] FAISS Vector Index (in-memory similarity search)
    ↓
User Question → Embed Query → [5] Similarity Search (Top-K chunks)
    ↓
[6] ConversationalRetrievalChain (Gemini 2.0-flash + Memory)
    ↓
Grounded Answer + Source Documents
```

**Why RAG over direct LLM prompting:**
- LLMs have context window limits — RAG retrieves only relevant sections
- Reduces hallucination — answers are grounded in actual document content
- Enables Q&A over arbitrarily long documents
- Source attribution verifies which parts of the PDF informed each answer

---

## Core Components

### 1. PDF Text Extraction

```python
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
```

**PyPDF2** reads PDF page-by-page and extracts raw text. Multiple PDFs are processed sequentially, concatenating all text into a single string for downstream chunking.

---

### 2. Text Chunking

```python
from langchain.text_splitter import CharacterTextSplitter

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap between consecutive chunks
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
```

**Why chunking:**
- Embedding models have token limits (~8K tokens for most)
- Smaller focused chunks yield better semantic similarity matching
- Overlapping ensures context continuity at chunk boundaries

**Chunking parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `chunk_size` | 1000 chars | Each chunk ≈ 100–200 words |
| `chunk_overlap` | 200 chars | 20% overlap prevents boundary cut-offs |
| `separator` | `\n` | Prefer paragraph boundaries |

---

### 3. Vector Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

**Google's `embedding-001`** converts each text chunk into a dense vector in a high-dimensional semantic space. Similar text → similar vectors → close proximity in vector space.

**Embedding process:**

$$\text{chunk} \xrightarrow{\text{Gemini embedding-001}} \vec{v} \in \mathbb{R}^{768}$$

**Cosine similarity** is used to find the most relevant chunks for a query:

$$\text{similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}||\vec{d}|}$$

---

### 4. FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
```

**FAISS (Facebook AI Similarity Search)** is an in-memory vector database optimized for fast approximate nearest-neighbor search. It indexes all chunk embeddings and retrieves the top-K most similar chunks for any query.

**FAISS advantages:**
- Sub-millisecond retrieval over millions of vectors
- Pure in-memory — no database setup required
- Supports cosine and L2 distance metrics

---

### 5. Conversational Q&A Chain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    retriever = vector_store.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        return_source_documents=True,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain
```

**Chain components:**

| Component | Implementation | Role |
|-----------|---------------|------|
| LLM | `gemini-2.0-flash` | Generates final answer from retrieved context |
| Retriever | FAISS similarity search | Fetches top-K relevant document chunks |
| Memory | `ConversationBufferMemory` | Stores full conversation history for follow-up questions |
| Chain | `ConversationalRetrievalChain` | Orchestrates retrieval → context injection → LLM response |

**Query handling flow:**

```python
def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    
    answer           = response.get('answer', None)
    source_documents = response.get('source_documents', None)
    
    st.session_state.answer          = answer
    st.session_state.source_documents = source_documents
    st.session_state.chat_history    = response['chat_history']
    
    # Display source documents in expandable section
    if source_documents:
        with st.expander("Source Documents"):
            st.write(source_documents)
```

---

## Streamlit UI

The app uses a custom chat interface defined in `chat_ui.py`:

```python
# chat_ui.py provides:
# - css: custom styling for the chat window
# - user_template: HTML template for user messages
# - bot_template: HTML template for bot responses
```

**Main UI layout:**

```python
st.set_page_config(page_title="PDF Data analyzer - using RAG.", page_icon=":books:")
st.header("PDF Data analyzer - using RAG.")

# Sidebar — PDF upload and processing
with st.sidebar:
    st.subheader("Your PDFs")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'",
        type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text    = "".join([get_pdf_text(pdf) for pdf in pdf_docs])
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_store)

# Main area — Q&A
user_question = st.text_input("Ask a question about the inputted PDFs:")
if user_question:
    handle_user_question(user_question)
```

---

## AWS S3 Integration

PDFs are optionally persisted to **AWS S3** for cloud storage:

```python
import boto3

def save_file_to_s3(file):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("aws_access_key"),
        aws_secret_access_key=os.getenv("aws_secret_access_key")
    )
    file.seek(0)
    s3.upload_fileobj(file, os.getenv("aws_bucket"), file.name)
```

AWS credentials are loaded from environment variables (`.env` file via `python-dotenv`).

---

## Environment Configuration

Create a `.env` file in the project root (never commit this):

```env
# Google AI (Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# AWS S3 (optional — for PDF storage)
aws_access_key=your_aws_access_key_id
aws_secret_access_key=your_aws_secret_access_key
aws_bucket=your_s3_bucket_name
```

**Getting API keys:**
- **Google Gemini:** [Google AI Studio](https://aistudio.google.com/) → Create API Key
- **AWS:** IAM Console → Create User → Attach S3FullAccess policy → Generate access keys

---

## Tech Stack

| Library | Version | Usage |
|---------|---------|-------|
| Python | 3.x | Core language |
| Streamlit | 1.x | Web UI framework |
| LangChain | 0.x | RAG orchestration, chain management |
| langchain-community | 0.x | FAISS vector store integration |
| langchain-google-genai | 0.x | Google Gemini embedding + LLM integration |
| Google Gemini | 2.0-flash | LLM for answer generation (fast, low-cost) |
| Google embedding-001 | — | Text-to-vector embedding model |
| FAISS (faiss-cpu) | 1.x | In-memory vector similarity search |
| PyPDF2 | 3.x | PDF text extraction |
| boto3 | 1.x | AWS S3 SDK for Python |
| python-dotenv | 1.x | Environment variable management |
| tiktoken | 0.x | Token counting (for context window management) |

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/nithinrajkore/PDF-DataAnalyzer.git
cd PDF-DataAnalyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env
# → Edit .env and add your GOOGLE_API_KEY

# 4. Run the app
streamlit run app.py
# → Open http://localhost:8501 in your browser

# 5. Usage
# - Upload one or more PDFs in the sidebar
# - Click "Process"
# - Type your question in the main text box
# - Expand "Source Documents" to see which PDF sections were used
```

**requirements.txt:**
```
streamlit
python-dotenv
PyPDF2
langchain
boto3
tiktoken
faiss-cpu
langchain-community
langchain_google_genai
```

---

## Key Concepts Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation: combine LLM generation with retrieved relevant documents |
| **Embedding** | Dense vector representation of text capturing semantic meaning |
| **Vector Store** | Database storing embeddings; supports similarity search |
| **FAISS** | Facebook AI Similarity Search — fast in-memory nearest-neighbor search |
| **Chunk** | A fixed-size text segment with optional overlap for granular retrieval |
| **Chunk Overlap** | Shared text between consecutive chunks; preserves context at boundaries |
| **Semantic Search** | Finding relevant documents by vector similarity, not keyword matching |
| **LangChain** | Python framework orchestrating LLMs, memory, chains, and retrievers |
| **ConversationBufferMemory** | LangChain memory that stores the full conversation history |
| **ConversationalRetrievalChain** | LangChain chain combining retrieval and conversational Q&A |
| **Gemini 2.0-flash** | Google's fast, cost-efficient LLM variant for production workloads |
| **Streamlit** | Python library for building data apps with no frontend code |
| **python-dotenv** | Loads environment variables from a `.env` file |

---

## References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
2. LangChain documentation: [https://python.langchain.com/](https://python.langchain.com/)
3. Google Gemini API: [https://ai.google.dev/](https://ai.google.dev/)
4. FAISS: [https://faiss.ai/](https://faiss.ai/)
5. Streamlit documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
6. PyPDF2: [https://pypdf2.readthedocs.io/](https://pypdf2.readthedocs.io/)
