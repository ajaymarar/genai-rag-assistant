# GenAI Enterprise Knowledge Assistant (RAG)

This project is a Retrieval Augmented Generation (RAG) system that allows users to query multiple enterprise documents using a local, open-source GenAI pipeline.

## Features
- Loads multiple PDF documents automatically
- Splits documents into semantic chunks
- Generates embeddings using HuggingFace sentence-transformers
- Stores embeddings in FAISS vector database
- Uses a local CPU-friendly LLM (FLAN-T5)
- Provides context-aware answers with source citations
- No paid APIs required

## Tech Stack
- Python
- LangChain
- HuggingFace Transformers
- FAISS
- Sentence-Transformers

## How it works
1. PDFs are loaded from the `data/` folder
2. Text is chunked and embedded
3. Relevant chunks are retrieved using vector similarity
4. The LLM generates answers grounded in retrieved context
5. Source documents and page numbers are displayed

## Run locally
```bash
python app.py
