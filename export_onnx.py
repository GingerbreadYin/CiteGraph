"""Export the existing BGE encoder with its CLS pooling and normalization."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class BgeEmbedding(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, attention_mask, token_type_ids):
        tokens = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, return_dict=False)[0]
        return functional.normalize(tokens[:, 0, :], p=2, dim=1)


def export(model, output):
    transformer = model[0]
    pooling = model[1]
    if (len(model) != 3 or type(model[2]).__name__ != "Normalize"
            or pooling.pooling_mode != "cls" or model.max_seq_length != 512):
        raise ValueError("Unexpected BGE SentenceTransformer configuration")
    inputs = ("input_ids", "attention_mask", "token_type_ids")
    tokenizer = transformer.tokenizer
    features = tokenizer(["导出用示例", "较长的中文测试句子"], padding=True, return_tensors="pt")
    with torch.inference_mode():
        torch.onnx.export(
            BgeEmbedding(transformer.auto_model).eval(), tuple(features[key] for key in inputs),
            str(output), input_names=list(inputs), output_names=["embedding"],
            dynamic_axes={**{name: {0: "batch", 1: "sequence"} for name in inputs},
                          "embedding": {0: "batch"}},
            dynamo=False, opset_version=17,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / ".model-cache")
    args = parser.parse_args()
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(args.cache_dir),
                                local_files_only=True, model_kwargs={"attn_implementation": "eager"})
    export(model, args.output)


if __name__ == "__main__":
    main()
