import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
import os


HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_PATH = "campbell_qa.jsonl"
TOP_K = [1, 2, 3]

class SpladeWrapper:
    def __init__(self, model_id: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _encode(self, texts: list[str]) -> list[dict[int, float]]:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs.logits
            # فرمول SPLADE: ReLU + Log-saturation + Max pooling
            relu_log = torch.log(1 + torch.relu(logits))
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            sparse_vecs = torch.max(relu_log * attention_mask, dim=1).values

        results = []
        for vec in sparse_vecs:
            nonzero_indices = torch.nonzero(vec).squeeze(-1)
            nonzero_values = vec[nonzero_indices]
            sparse_dict = {
                idx.item(): val.item() 
                for idx, val in zip(nonzero_indices, nonzero_values)
            }
            results.append(sparse_dict)
        return results

    def embed_query(self, query: str) -> dict[int, float]:
        return self._encode([query])[0]

    def embed_docs(self, docs: list[str]) -> list[dict[int, float]]:
        return self._encode(docs)


class BGEM3SparseWrapper:
    def __init__(self, model_id: str = "BAAI/bge-m3", device: str = None):
        use_fp16 = torch.cuda.is_available()
        self.model = BGEM3FlagModel(model_id, use_fp16=use_fp16, device=device)

    def embed_query(self, query: str) -> dict[str, float]:
        out = self.model.encode([query], return_dense=False, return_sparse=True)
        return out["lexical_weights"][0]

    def embed_docs(self, docs: list[str]) -> list[dict[str, float]]:
        out = self.model.encode(docs, return_dense=False, return_sparse=True)
        return out["lexical_weights"]


def sparse_dot_product_dict(query_dict: dict, doc_dict: dict) -> float:
    score = 0.0
    for token_id, weight in query_dict.items():
        if token_id in doc_dict:
            score += weight * doc_dict[token_id]
    return float(score)

def recall_at_k(ranked_indices, relevant_index, k):
    return float(relevant_index in ranked_indices[:k])

def reciprocal_rank_at_k(ranked_indices, relevant_index, k):
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx == relevant_index:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked_indices, relevant_index, k):
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx == relevant_index:
            return 1.0 / np.log2(rank + 1)
    return 0.0

def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def prepare_dataset(data):
    queries = []
    candidate_documents = []
    relevant_indices = []

    for item in data:
        question = item.get("question", "").strip()
        options = item.get("options", {})
        correct_answer = item.get("answer", "").strip().lower()

        if not question or not isinstance(options, dict) or correct_answer not in options:
            continue

        queries.append(question)
        docs = []
        correct_index = None

        for idx, (label, option_text) in enumerate(options.items()):
            docs.append(str(option_text).strip())
            if label.lower() == correct_answer:
                correct_index = idx

        candidate_documents.append(docs)
        relevant_indices.append(correct_index)

    return queries, candidate_documents, relevant_indices

def evaluate_model_pipeline(model_name, model_instance, queries, candidate_documents, relevant_indices):
    print("\n" + "=" * 70)
    print(f"Evaluating Model: {model_name}")
    print("=" * 70)

    metrics = {f"{metric}@{k}": [] for metric in ["Recall", "MRR", "nDCG"] for k in TOP_K}

    for query, docs, relevant_idx in tqdm(zip(queries, candidate_documents, relevant_indices), total=len(queries)):
        query_sparse = model_instance.embed_query(query)
        docs_sparse = model_instance.embed_docs(docs)

        scores = [sparse_dot_product_dict(query_sparse, d_sparse) for d_sparse in docs_sparse]
        ranked_indices = np.argsort(scores)[::-1]

        for k in TOP_K:
            metrics[f"Recall@{k}"].append(recall_at_k(ranked_indices, relevant_idx, k))
            metrics[f"MRR@{k}"].append(reciprocal_rank_at_k(ranked_indices, relevant_idx, k))
            metrics[f"nDCG@{k}"].append(ndcg_at_k(ranked_indices, relevant_idx, k))

    result = {"model": model_name, "num_questions": len(queries)}
    for metric_name, values in metrics.items():
        result[metric_name] = np.mean(values)

    return result

def main():
    data = load_dataset(DATASET_PATH)
    queries, candidate_documents, relevant_indices = prepare_dataset(data)

    models_to_test = [
        ("naver/splade_v2_max", SpladeWrapper("naver/splade_v2_max")),
        ("BAAI/bge-m3 (Lexical)", BGEM3SparseWrapper("BAAI/bge-m3")),
    ]

    results = []
    for model_name, model_inst in models_to_test:
        res = evaluate_model_pipeline(model_name, model_inst, queries, candidate_documents, relevant_indices)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values(by="MRR@1", ascending=False)

    print("\n" + "=" * 100)
    print("FINAL RESULTS (ADVANCED SPARSE MODELS)")
    print("=" * 100)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv("campbell_advanced_sparse_results.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
