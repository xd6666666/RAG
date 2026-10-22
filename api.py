
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from task2_rag.retrieval.retriever import Retriever
from task2_rag.generation.llm import QwenLLM


app = FastAPI(
    title="Insurance RAG API",
    description="RAG system based on BGE-M3, Milvus and Qwen",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        # 1. Retrieve relevant documents from Milvus
        retriever = Retriever()
        results = retriever.search(
            request.question,
            top_k=2
        )

        # 2. Extract document contexts
        contexts = [
            item["content"]
            for item in results
        ]

        # 3. Generate answer with Qwen
        llm = QwenLLM()
        answer = llm.chat(
            request.question,
            contexts
        )

        # 4. Return JSON response
        return QueryResponse(
            question=request.question,
            answer=answer,
            contexts=contexts
        )

    except Exception as e:
        print(f"RAG error: {e}")
        raise HTTPException(
            status_code=500,
            detail="RAG processing failed. Check server logs."
        )

