import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastembed import SparseTextEmbedding

DATASET_PATH = "campbell_qa.jsonl"

# Sparse models
MODELS = [
    "prithivida/Splade_PP_en_v1",
    "Qdrant/bm42-all-minilm-l6-v2-attentions",
]

TOP_K = [1, 2, 3]

def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON error in line {line_number}: {e}")
    print(f"Loaded {len(data)} questions.")
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
            option_text = str(option_text).strip()
            document = option_text 
            docs.append(document)
            
            if label.lower() == correct_answer:
                correct_index = idx

        candidate_documents.append(docs)
        relevant_indices.append(correct_index)

    print(f"Prepared {len(queries)} benchmark queries.")
    return queries, candidate_documents, relevant_indices

def sparse_dot_product(query_embedding, doc_embedding):
    query_indices = query_embedding.indices
    query_values = query_embedding.values
    doc_indices = doc_embedding.indices
    doc_values = doc_embedding.values

    doc_dict = dict(zip(doc_indices, doc_values))
    score = 0.0

    for idx, value in zip(query_indices, query_values):
        if idx in doc_dict:
            score += value * doc_dict[idx]
    return score

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

def evaluate_model(model_name, queries, candidate_documents, relevant_indices):
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    model = SparseTextEmbedding(model_name=model_name)
    metrics = {f"{metric}@{k}": [] for metric in ["Recall", "MRR", "nDCG"] for k in TOP_K}

    for query, docs, relevant_idx in tqdm(zip(queries, candidate_documents, relevant_indices), total=len(queries), desc="Evaluating"):
        query_embedding = list(model.query_embed(query))[0]
        doc_embeddings = list(model.embed(docs))

        scores = []
        for doc_embedding in doc_embeddings:
            score = sparse_dot_product(query_embedding, doc_embedding)
            scores.append(score)
            
        scores = np.array(scores)
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

    if not queries:
        raise RuntimeError("No benchmark queries were prepared. Check the dataset structure.")

    results = []
    for model_name in MODELS:
        result = evaluate_model(model_name, queries, candidate_documents, relevant_indices)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="MRR@1", ascending=False) # Changed to MRR@1 as it makes more sense for MCQs

    print("\n" + "=" * 100)
    print("FINAL SPARSE RETRIEVAL RESULTS")
    print("=" * 100)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_path = "campbell_sparse_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
