# GenAI RAG Assistant – Architecture

## Overview
This project implements a Retrieval Augmented Generation (RAG) system exposed via a FastAPI service.

## Architecture Flow

Client (Browser / API Consumer)
        ↓
FastAPI (`/ask` endpoint)
        ↓
RAG Core (`rag_core.py`)
        ↓
FAISS Vector Store (local)
        ↓
Relevant Document Chunks
        ↓
Local LLM (FLAN-T5)
        ↓
Answer + Source Citations

## Key Design Decisions
- Local open-source models (no paid APIs)
- FAISS for fast semantic retrieval
- Prompt grounding to reduce hallucinations
- Source citations for enterprise traceability

## Limitations
- Uses a lightweight instruction model (FLAN-T5-small)
- Binary-style questions may result in categorical answers
- Designed for demonstration and learning, not financial advice

## Future Improvements
- Swap to stronger local LLM (Mistral / LLaMA)
- Add re-ranking for better retrieval
- Add evaluation & monitoring
