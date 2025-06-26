# 🧠 Agentic University QA System with RAG + LLMs

This project implements a modular **information retrieval and question-answering (QA) system** that combines **Retrieval-Augmented Generation (RAG)** and **instruction-tuned large language models (LLMs)** to answer user queries grounded in real-world university website content.

The system scrapes webpages from the [University of Memphis](https://www.memphis.edu) domain, indexes them using dense vector representations, and generates answers using either **LLaMA 3.1-8B Instruct** or **Mistral-7B Instruct**. A Streamlit interface allows dynamic switching between LLMs and two generation modes: **RAG-based QA** or **direct (no-RAG) generation**.

---

## 🛠️ Architecture & Pipeline Overview

The system consists of the following modular stages:

### 1. **Web Scraping**
- URLs from the `memphis.edu` domain are scraped using `requests` and parsed with `BeautifulSoup`.
- HTML content is stripped of scripts, styles, and navigation to extract clean text content.
- Extracted documents are stored in memory as a list of raw strings.

### 2. **Semantic Embedding and Indexing**
- Text content is embedded using the `all-MiniLM-L6-v2` model from **SentenceTransformers**.
- All document embeddings are stored in a **FAISS IndexFlatL2** index to support fast nearest-neighbor search.
- The embeddings are computed only once and cached using `@st.cache_resource`.

### 3. **User Query Flow (RAG vs. No-RAG)**

#### 🔹 RAG Mode:
- User inputs a question.
- The query is embedded and used to retrieve top-k similar documents from the FAISS index.
- Retrieved documents are concatenated into a context block.


## 🔧 Features

- ✅ User-selectable LLMs: `LLaMA 3.1` or `Mistral-7B-Instruct`
- ✅ Choose between **RAG mode** (retrieves and grounds responses) or **No-RAG mode**
- ✅ Semantic search with `SentenceTransformers` and `FAISS`
- ✅ Clean dark-mode answer formatting with context expanders
- ✅ Scrapes real-time content from university domains
- ✅ Caching with `@st.cache_data` and `@st.cache_resource` for performance

---

## 📸 Demo

![demo](assets/image.png)