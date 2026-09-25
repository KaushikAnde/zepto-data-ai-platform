
# Zepto Support Assistant

## Overview

This module implements a small Retrieval-Augmented Generation (RAG)
support assistant for Zepto policies.

The system uses:

- Sentence Transformers for embeddings
- ChromaDB for vector storage
- LangGraph for query routing
- Pydantic for structured responses
- FastAPI for the REST API
- Uvicorn for serving the application

The default graded implementation uses `MOCK_LLM=1`, so no external
LLM API or API key is required.

---

## Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── ingest.py
├── prompt.py
├── graph.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
