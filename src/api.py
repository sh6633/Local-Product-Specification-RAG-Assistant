from threading import Lock

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm import LocalLLM
from main import FALLBACK_ANSWER
from rag_natural import NaturalRAGPipeline


app = FastAPI(title="AORUS MASTER 16 AM6H RAG API")

_rag: NaturalRAGPipeline | None = None
_llm: LocalLLM | None = None
_load_lock = Lock()
_generation_lock = Lock()


class QuestionRequest(BaseModel):
    question: str


def get_services() -> tuple[NaturalRAGPipeline, LocalLLM]:
    global _rag, _llm
    if _rag is not None and _llm is not None:
        return _rag, _llm

    with _load_lock:
        if _rag is None:
            _rag = NaturalRAGPipeline()
        if _llm is None:
            _llm = LocalLLM()
    return _rag, _llm


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: QuestionRequest) -> dict:
    rag, llm = get_services()
    chunks = rag.retrieve(request.question)
    if not chunks:
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved_chunks": [],
            "metrics": {"ttft": None, "tps": 0.0, "tokens": 0},
        }

    prompt = rag.build_prompt(request.question, chunks)
    answer_parts = []
    metrics = None

    with _generation_lock:
        for token, maybe_metrics in llm.stream(prompt):
            if token:
                answer_parts.append(token)
            if maybe_metrics:
                metrics = maybe_metrics

    return {
        "answer": "".join(answer_parts),
        "retrieved_chunks": chunks,
        "metrics": metrics,
    }


@app.post("/chat/stream")
def chat_stream(request: QuestionRequest) -> StreamingResponse:
    rag, llm = get_services()
    chunks = rag.retrieve(request.question)
    if not chunks:
        return StreamingResponse(iter([FALLBACK_ANSWER]), media_type="text/plain")

    prompt = rag.build_prompt(request.question, chunks)

    def generate():
        with _generation_lock:
            for token, _ in llm.stream(prompt):
                if token:
                    yield token

    return StreamingResponse(generate(), media_type="text/plain")
