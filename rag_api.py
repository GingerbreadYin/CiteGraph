from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_graph import build_rag_graph
from rag_service import make_result
from retrieval import load_retriever, with_parent_child_index
import logging
from citations import InvalidAnswerError
from evidence import InvalidEvidenceOutputError
from model_client import ModelRequestError

logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app):
    print("正在加载检索资源并构建图……")
    resources = load_retriever()
    if os.getenv("RAG_PARENT_CHILD_ENABLED") == "1":
        resources = with_parent_child_index(resources)
    retrieval_mode = os.getenv("RAG_RETRIEVAL_MODE", "vector")
    if retrieval_mode != "vector":
        from hybrid_retrieval import with_hybrid_search

        cache_dir = Path(os.environ.get("RAG_RERANKER_CACHE_DIR") or Path(__file__).parent / ".model-cache")
        resources = with_hybrid_search(resources, mode=retrieval_mode, cache_dir=cache_dir)
    app.state.rag_graph = build_rag_graph(
        resources,
        query_split_enabled=os.getenv("RAG_QUERY_SPLIT_ENABLED") == "1",
        query_rewrite_enabled=os.getenv("RAG_QUERY_REWRITE_ENABLED") == "1",
    )
    print("RAG 图准备完成")
    yield


app = FastAPI(title="电商政策 RAG API", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest):
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空白")

    try:
        state = app.state.rag_graph.invoke({"question": question})
    except ModelRequestError:
        logger.exception("处理问答时，模型请求失败")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_REQUEST_FAILED",
                "message": "模型服务当前不可用，本次未能完成处理。",
            },
        ) from None
    except InvalidEvidenceOutputError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "INVALID_EVIDENCE_OUTPUT",
                "message": "证据检查服务返回了无效结果，本次未能完成处理。",
            },
        ) from None
    except InvalidAnswerError:
        logger.exception("模型回答未通过引用校验")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "INVALID_MODEL_OUTPUT",
                "message": "本次回答未通过引用校验，暂未返回。",
            },
        ) from None
    return make_result(
        state["status"],
        state["answer"],
        state["rounds"],
        state["selected_chunks"],
        state.get("final_answer_usage"),
    )