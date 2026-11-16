import hashlib
import json
import os
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


def load_retriever():
    """读取资料、索引并加载模型，返回检索所需资源。"""
    project_dir = Path(__file__).parent
    source_bytes = (project_dir / "chunks.json").read_bytes()
    chunks = json.loads(source_bytes)

    index = torch.load(
        project_dir / "vector_index.pt",
        map_location="cpu",
        weights_only=True,
    )

    if index["source_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise RuntimeError("文档块已变化，请重新运行 build_index.py")

    model_name = "BAAI/bge-small-zh-v1.5"

    if index["model_name"] != model_name:
        raise RuntimeError("索引模型与问题向量模型不一致")

    model = SentenceTransformer(
        model_name,
        device="cpu",
        cache_folder=str(project_dir / ".model-cache"),
    )

    onnx_path = os.getenv("RAG_ONNX_MODEL_PATH")
    if onnx_path:
        from onnx_encoder import OnnxEncoder

        model = OnnxEncoder(onnx_path, model.tokenizer, model.max_seq_length)
    return {
        "chunks": chunks,
        "model": model,
        "vectors": index["vectors"],
    }


def with_parent_child_index(resources):
    """Build an in-memory sentence index pointing to the original policy sections."""
    children = []
    for parent_index, parent in enumerate(resources["chunks"]):
        text = parent["text"].strip()
        lines = text.splitlines()
        body = "\n".join(lines[1:]).strip() if lines and lines[0].startswith("## ") else text
        sentences = re.split(r"(?<=[。！？；])\s*|\n+", body)
        for sentence in (part.strip() for part in sentences if part.strip()):
            children.append((parent_index, f"{parent['section']}\n{sentence}"))
        if not body:
            children.append((parent_index, f"{parent['section']}\n{text}"))
    if not children:
        return resources
    vectors = resources["model"].encode([text for _, text in children], convert_to_tensor=True)
    return {**resources, "search_chunks": children, "vectors": vectors}


def rank_chunks(question, resources):
    """计算相关性，返回从高到低排列的完整文档块记录。"""
    query = "为这个句子生成表示以用于检索相关文章：" + question
    query_vector = resources["model"].encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vector, resources["vectors"])[0]
    indices = scores.argsort(descending=True)

    if "search_chunks" in resources:
        children = resources["search_chunks"]
        ranked = list(dict.fromkeys(children[int(index)][0] for index in indices))
    else:
        ranked = [int(index) for index in indices]

    if "bm25" in resources:
        from hybrid_retrieval import rank_hybrid

        ranked = rank_hybrid(question, ranked, resources)
    return [resources["chunks"][index] for index in ranked]


def format_evidence(chunks):
    """将文档块整理成带编号、来源和章节的参考资料。"""
    return "\n".join(
        f"[{chunk['id']}]\n"
        f"来源：{chunk['source']}，章节：{chunk['section']}\n"
        f"{chunk['text']}"
        for chunk in chunks
    )