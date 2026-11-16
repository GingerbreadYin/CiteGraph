"""Optional CPU backend for the exported BGE sentence encoder."""
import torch


class OnnxEncoder:
    def __init__(self, model_path, tokenizer, max_length, *, threads=1):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        if set(self.input_names) != {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("Unexpected BGE ONNX inputs")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, texts, *, convert_to_tensor=False):
        single = isinstance(texts, str)
        batch = [texts] if single else texts
        tokens = self.tokenizer(
            batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="np",
        )
        vectors = self.session.run(["embedding"], {name: tokens[name] for name in self.input_names})[0]
        if single:
            vectors = vectors[0]
        return torch.from_numpy(vectors) if convert_to_tensor else vectors
