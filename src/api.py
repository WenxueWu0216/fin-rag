from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import bm25_search, dense_search, hybrid_search, rerank_search
from rag import answer

METHODS = {
    "bm25": bm25_search.search,
    "dense": dense_search.search,
    "hybrid": hybrid_search.search,
    "rerank": rerank_search.search,
}

app = FastAPI(title="fin-rag", version="0.1.0")


class Hit(BaseModel):
    chunk_id: str
    score: float
    ticker: str
    title: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=10, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=list[Hit])
def search(
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=30),
    method: Literal["bm25", "dense", "hybrid", "rerank"] = "rerank",
):
    results = METHODS[method](q, k=k)
    return [Hit(chunk_id=r[0], score=r[1], ticker=r[2], title=r[3]) for r in results]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        out = answer(req.question, k=req.k)
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}")
    return AskResponse(
        answer=out["answer"],
        citations=out["citations"],
        invalid_citations=out["invalid_citations"],
        sources=out["sources"],
    )