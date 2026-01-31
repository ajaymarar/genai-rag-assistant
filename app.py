from fastapi import FastAPI
from pydantic import BaseModel

from rag_core import answer_question

app = FastAPI(title="GenAI RAG Assistant")

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Health check
@app.get("/")
def health_check():
    return {"status": "FastAPI is running"}

# RAG endpoint
@app.post("/ask")
def ask_question(payload: QuestionRequest):
    result = answer_question(payload.question)

    return {
        "answer": result["result"],
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in result["source_documents"]
        ]
    }
