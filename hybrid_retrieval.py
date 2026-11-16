"""Optional BM25, reciprocal rank fusion and local cross-encoder reranking."""
from math import isfinite

import jieba
from rank_bm25 import BM25Okapi


CANDIDATE_DEPTH = 6
RRF_K = 60


def tokenize(text):
    return [word.casefold() for word in jieba.lcut(text) if word.isalnum()]


def with_hybrid_search(resources, *, mode, cache_dir=None, local_files_only=False):
    if mode not in {"hybrid", "rerank"}:
        raise ValueError("RAG_RETRIEVAL_MODE must be vector, hybrid, or rerank")

    corpus = [tokenize(chunk["text"]) for chunk in resources["chunks"]]
    index = BM25Okapi(corpus) if any(corpus) else None
    configured = {**resources, "bm25": index}
    if mode == "rerank":
        from sentence_transformers import CrossEncoder

        configured["reranker"] = CrossEncoder(
            "BAAI/bge-reranker-base", device="cpu", max_length=512,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
    return configured


def fuse_rankings(dense, lexical):
    scores = {}
    for ranking in (dense, lexical):
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0) + 1 / (RRF_K + rank)
    # Stable sorting gives vector candidates priority when fused scores tie.
    return sorted(scores, key=scores.__getitem__, reverse=True)


def rank_hybrid(question, dense, resources):
    index = resources["bm25"]
    tokens = tokenize(question)
    lexical = []
    if index is not None and tokens:
        scores = index.get_scores(tokens)
        lexical = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])
                   if scores[i] > 0][:CANDIDATE_DEPTH]

    fused = fuse_rankings(dense[:CANDIDATE_DEPTH], lexical)
    reranker = resources.get("reranker")
    if reranker is not None and fused:
        head = fused[:CANDIDATE_DEPTH]
        pairs = [(question, resources["chunks"][i]["text"]) for i in head]
        scores = [float(score) for score in reranker.predict(pairs)]
        if len(scores) != len(head) or not all(isfinite(score) for score in scores):
            raise RuntimeError("Cross-Encoder returned invalid relevance scores")
        ranked = sorted(zip(head, scores), key=lambda item: -item[1])
        fused = [i for i, _ in ranked] + fused[CANDIDATE_DEPTH:]
    seen = set(fused)
    return fused + [i for i in dense if i not in seen]
