"""Optional V1 fallback for the RAG graph's first evidence failure."""
from http.client import HTTPException
import json
import logging
import re

from model_client import ModelRequestError, call_deepseek


logger = logging.getLogger(__name__)
CONTEXT_CHARS = 1800
SPLIT_INSTRUCTIONS = """你把电商政策问题拆成两条独立检索查询。保留用户已明确给出的对象、条件、否定和范围，子查询要有足够上下文。不要回答问题，不得添加用户没说明的事实，不得猜测政策条款，不要改变问题的时间顺序或条件。如果无法安全拆成两条，返回空数组。只输出 JSON 对象，结构是 {"subqueries":["第一条查询","第二条查询"]} 或 {"subqueries":[]}。"""


def split_queries(question):
    """Return two usable subqueries, or no queries so the caller can fall back."""
    try:
        response = call_deepseek({
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": SPLIT_INSTRUCTIONS},
                {"role": "user", "content": question},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 240,
            "temperature": 0,
        }, max_attempts=1)
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            return []
        result = json.loads(choice["message"]["content"])
    except (ModelRequestError, HTTPException, OSError, KeyError, IndexError, TypeError, ValueError):
        logger.warning("查询拆分未返回可用结果，使用原排序补检")
        return []

    if not isinstance(result, dict) or set(result) != {"subqueries"}:
        return []
    queries = result["subqueries"]
    if not isinstance(queries, list) or len(queries) != 2:
        return []
    if any(not isinstance(query, str) or not query.strip() or len(query) > 160 for query in queries):
        return []

    queries = [query.strip() for query in queries]
    normalized = [re.sub(r"[\W_]+", "", query.casefold()) for query in queries]
    if not all(normalized) or normalized[0] == normalized[1]:
        return []
    return queries


def expand_with_query_split(question, ranked_chunks, retrieve):
    """Keep the first round, add split candidates, then fill with original hits."""
    baseline = ranked_chunks[:4]
    first = ranked_chunks[:2]
    remaining = CONTEXT_CHARS - sum(len(chunk["text"]) for chunk in first)
    if remaining <= 0:
        return baseline
    queries = split_queries(question)
    if not queries:
        return baseline

    try:
        child_hits = [retrieve(query)[:2] for query in queries]
    except (RuntimeError, ValueError, OSError):
        logger.warning("子查询检索失败，使用原排序补检")
        return baseline

    original_positions = {chunk["id"]: rank for rank, chunk in enumerate(first)}
    entries = [
        (rank, branch, chunk)
        for branch, hits in enumerate(child_hits)
        for rank, chunk in enumerate(hits)
    ]
    entries.sort(key=lambda entry: (
        entry[0], original_positions.get(entry[2]["id"], len(first)), entry[1],
    ))

    selected = list(first)
    seen = {chunk["id"] for chunk in first}
    for chunk in [entry[2] for entry in entries] + baseline:
        length = len(chunk["text"])
        if chunk["id"] in seen or length > remaining:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        remaining -= length
        if len(selected) == 4:
            break

    # Keep the baseline order when splitting has not provided any additional evidence.
    baseline_ids = {chunk["id"] for chunk in baseline}
    return selected if seen - baseline_ids else baseline
