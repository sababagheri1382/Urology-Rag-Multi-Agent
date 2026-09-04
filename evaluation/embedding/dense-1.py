import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = "campbell_qa.jsonl"

MODELS = [
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/all-mpnet-base-v2"
]

TOP_K = [1, 3, 5, 10]

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
        answer_text = item.get("answer_text", "").strip()
        explanation = item.get("explanation", "").strip()
        if not question:
            continue
        if not isinstance(options, dict):
            continue
        if correct_answer not in options:
            print(
                f"Warning: invalid answer "
                f"for {item.get('id')}"
            )
            continue

        queries.append(question)
        docs = []
        correct_index = None

        for idx, (label, option_text) in enumerate(options.items()):
            option_text = str(option_text).strip()
            # Correct answer
            if label.lower() == correct_answer:
                document = (
                    f"Answer: {option_text}\n"
                    f"Explanation: {explanation}"
                )
                correct_index = idx
            # Incorrect answer
            else:
                document = option_text
            docs.append(document)
        candidate_documents.append(docs)
        relevant_indices.append(correct_index)
    print(f"Prepared {len(queries)} benchmark queries.")
    return (queries, candidate_documents, relevant_indices)

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
    model = SentenceTransformer(model_name)

    metrics = {}
    for k in TOP_K:
        metrics[f"Recall@{k}"] = []
        metrics[f"MRR@{k}"] = []
        metrics[f"nDCG@{k}"] = []

    for query, docs, relevant_idx in tqdm(zip(queries, candidate_documents, relevant_indices), total=len(queries), desc="Evaluating"):
        query_embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        doc_embeddings = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
        scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]
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
    (queries, candidate_documents, relevant_indices) = prepare_dataset(data)
    if len(queries) == 0:
        raise RuntimeError(
            "No benchmark queries were prepared. "
            "Check the dataset structure."
        )

    results = []
    for model_name in MODELS:
        result = evaluate_model(model_name=model_name, queries=queries, candidate_documents=candidate_documents, relevant_indices=relevant_indices)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="MRR@5", ascending=False)

    print("\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv("campbell_embedding_results.csv", index=False, encoding="utf-8-sig")
    print(
        "\nResults saved to: "
        "campbell_embedding_results.csv"
    )

if __name__ == "__main__":
    main()