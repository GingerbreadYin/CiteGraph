"""Optional search-only rewrite for the second retrieval round."""
from http.client import HTTPException
import json
import logging
import re

from model_client import ModelRequestError, call_deepseek
from query_split import CONTEXT_CHARS


logger = logging.getLogger(__name__)


def rewrite_query(question):
    instructions = """把电商政策问题改写成一条适合检索的中文查询。用政策常用词表达口语，保留原有对象、事实、时间、否定和不确定性，不要回答问题。不得把原因未知写成已确认质量问题，不得省略多项问题中的任意一项。不能安全改写时返回空字符串。只输出 JSON 对象，结构为 {"query":"改写后的查询"}。"""
    try:
        response = call_deepseek({
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 240,
            "temperature": 0,
        }, max_attempts=1)
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            return ""
        result = json.loads(choice["message"]["content"])
    except (ModelRequestError, HTTPException, OSError, KeyError, IndexError, TypeError, ValueError):
        logger.warning("查询改写失败，使用原排序补检")
        return ""

    if not isinstance(result, dict) or set(result) != {"query"} or not isinstance(result["query"], str):
        return ""
    query = result["query"].strip()
    normalize = lambda text: re.sub(r"[\W_]+", "", text.casefold())
    if len(query) > 240 or normalize(query) in {"", normalize(question)}:
        return ""
    return query


def expand_with_query_rewrite(question, ranked_chunks, retrieve):
    baseline = ranked_chunks[:4]
    selected = baseline[:2]
    remaining = CONTEXT_CHARS - sum(len(chunk["text"]) for chunk in selected)
    if remaining <= 0:
        return baseline
    query = rewrite_query(question)
    if not query:
        return baseline
    try:
        rewritten = retrieve(query)[:2]
    except (RuntimeError, ValueError, OSError):
        logger.warning("改写查询检索失败，使用原排序补检")
        return baseline

    baseline_ids = {chunk["id"] for chunk in baseline}
    seen = {chunk["id"] for chunk in selected}
    for chunk in [*rewritten, *baseline]:
        size = len(chunk["text"])
        if chunk["id"] in seen or size > remaining:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        remaining -= size
        if len(selected) == 4:
            break
    return selected if seen - baseline_ids else baseline
