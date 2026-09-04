import os
import glob
import json
import random
import csv
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder

def load_xvx_dataset(folder_path="evaluation_datasets"):
    dataset = []
    file_pattern = os.path.join(folder_path, "*.json")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No JSON files found in '{folder_path}'.")
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for item in data:
                    if "question" in item and "answer_text" in item:
                        dataset.append({
                            "query": item["question"],
                            "context": item["answer_text"]
                        })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path}.")
                
    return dataset

def evaluate_multiple_rerankers(target_negatives=49):
    models_to_test = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2", 
        "BAAI/bge-reranker-base",  
        "cross-encoder/nli-deberta-v3-small"                   
    ]
    
    dataset = load_xvx_dataset("evaluation_datasets") 
    total_queries = len(dataset)
    
    if total_queries == 0:
        print("Dataset is empty. Exiting...")
        return
        
    num_negatives = min(target_negatives, total_queries - 1)
    print(f"Total Questions: {total_queries} | Noise Docs: {num_negatives}")
    
    summary_results = []
    
    for model_name in models_to_test:
        print(f"\n{'-'*50}\nLoading & Evaluating Model: {model_name}\n{'-'*50}")
        safe_model_name = model_name.replace("/", "_")
        
        try:
            reranker = CrossEncoder(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}. Error: {e}")
            continue
        
        ranks = []
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = 0, 0, 0, 0
        detailed_csv_data = []
        
        for i, item in enumerate(tqdm(dataset, desc=f"Evaluating")):
            query = item["query"]
            correct_context = item["context"]
            
            all_other_contexts = [dataset[j]["context"] for j in range(total_queries) if j != i]
            negative_contexts = random.sample(all_other_contexts, num_negatives)
            
            candidates = negative_contexts + [correct_context]
            random.shuffle(candidates)
            correct_index = candidates.index(correct_context)
            
            pairs = [[query, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            
            ranked_indices = np.argsort(scores)[::-1]
            rank = np.where(ranked_indices == correct_index)[0][0] + 1
            
            ranks.append(rank)
            
            is_hit_1 = (rank == 1)
            is_hit_3 = (rank <= 3)
            is_hit_5 = (rank <= 5)
            is_hit_10 = (rank <= 10)
            
            if is_hit_1: hits_at_1 += 1
            if is_hit_3: hits_at_3 += 1
            if is_hit_5: hits_at_5 += 1
            if is_hit_10: hits_at_10 += 1
                
            detailed_csv_data.append({
                "Query_ID": i + 1,
                "Query": query,
                "Rank": rank,
                "Reciprocal_Rank": round(1.0 / rank, 4),
                "Hit@1": int(is_hit_1),
                "Hit@3": int(is_hit_3),
                "Hit@5": int(is_hit_5),
                "Hit@10": int(is_hit_10)
            })

        mrr = np.mean([1.0 / r for r in ranks])
        hit1_pct = (hits_at_1 / total_queries) * 100
        hit3_pct = (hits_at_3 / total_queries) * 100
        hit5_pct = (hits_at_5 / total_queries) * 100
        hit10_pct = (hits_at_10 / total_queries) * 100
        
        detailed_csv_filename = f"details_{safe_model_name}.csv"
        with open(detailed_csv_filename, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Query_ID", "Query", "Rank", "Reciprocal_Rank", "Hit@1", "Hit@3", "Hit@5", "Hit@10"])
            writer.writeheader()
            writer.writerows(detailed_csv_data)
            
        summary_results.append({
            "Model_Name": model_name,
            "MRR": round(mrr, 4),
            "Hit@1 (%)": round(hit1_pct, 2),
            "Hit@3 (%)": round(hit3_pct, 2),
            "Hit@5 (%)": round(hit5_pct, 2),
            "Hit@10 (%)": round(hit10_pct, 2)
        })

    print("\nSaving Summary Results...")
    with open("summary_all_models.csv", mode='w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ["Model_Name", "MRR", "Hit@1 (%)", "Hit@3 (%)", "Hit@5 (%)", "Hit@10 (%)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)
        
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    evaluate_multiple_rerankers()
