import json
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir)
    potential_input_file = os.path.join(PROJECT_ROOT, "Retriever-Test", "relevance_evaluation_final.json")
    if not os.path.exists(potential_input_file):
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        INPUT_FILE = os.path.join(PROJECT_ROOT, "Retriever-Test", "relevance_evaluation_final.json")
        if not os.path.exists(INPUT_FILE):
            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            INPUT_FILE = os.path.join(PROJECT_ROOT, "Retriever-Test", "relevance_evaluation_final.json")
            if not os.path.exists(INPUT_FILE):
                raise FileNotFoundError("INPUT_FILE not found. Please check PROJECT_ROOT and INPUT_FILE path.")
    else:
        INPUT_FILE = potential_input_file

except NameError:
    PROJECT_ROOT = os.getcwd()
    INPUT_FILE = os.path.join(PROJECT_ROOT, "Retriever-Test", "relevance_evaluation_final.json")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("INPUT_FILE not found. Please ensure it exists in the current directory.")


K_VALUES = [1, 3, 5, 10, 15]

RELEVANCE_THRESHOLD = 3

def load_data():
    """Loads data from the specified JSON input file."""
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_FILE}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None


def get_failed_questions_at_k(data, k):
    failed_ids = []

    for item in data:
        evaluations = item.get("evaluations", [])
        evaluations = sorted(evaluations, key=lambda x: x["rank"])
        top_k = evaluations[:k]

        if not top_k:
            question_id = item.get("question_id", "Unknown ID")
            failed_ids.append(question_id)
            continue

        if not any(
            evaluation["relevance_score"] >= RELEVANCE_THRESHOLD
            for evaluation in top_k
        ):
            question_id = item.get("question_id", "Unknown ID")
            failed_ids.append(question_id)

    return failed_ids


def calculate_mean_relevance(data, k):
    total_relevance_sum_in_top_k = 0
    num_questions_with_evaluations = 0

    for item in data:
        evaluations = item.get("evaluations", [])
        evaluations = sorted(evaluations, key=lambda x: x["rank"])
        top_k = evaluations[:k]

        if not top_k:
            continue

        num_questions_with_evaluations += 1
        relevance_sum_for_item = sum(
            evaluation["relevance_score"] for evaluation in top_k
        )
        average_relevance_for_item = relevance_sum_for_item / len(top_k)
        total_relevance_sum_in_top_k += average_relevance_for_item

    if num_questions_with_evaluations == 0:
        return 0.0

    return total_relevance_sum_in_top_k / num_questions_with_evaluations


def calculate_relevance_hit(data, k):
    hits = 0
    valid_questions = 0
    failed_question_ids = []

    for item in data:
        evaluations = item.get("evaluations", [])
        evaluations = sorted(evaluations, key=lambda x: x["rank"])
        top_k = evaluations[:k]

        if not top_k:
            continue

        valid_questions += 1

        if any(
            evaluation["relevance_score"] >= RELEVANCE_THRESHOLD
            for evaluation in top_k
        ):
            hits += 1
        else:
            question_id = item.get("question_id", "Unknown ID")
            failed_question_ids.append(question_id)

    if valid_questions == 0:
        return 0.0, 0, 0, []

    hit_rate = hits / valid_questions

    return hit_rate, hits, valid_questions, failed_question_ids


def calculate_mrr(data, k):
    total_reciprocal_rank = 0
    valid_questions = 0

    for item in data:
        evaluations = item.get("evaluations", [])
        evaluations = sorted(evaluations, key=lambda x: x["rank"])
        top_k = evaluations[:k]

        if not top_k:
            continue

        valid_questions += 1

        reciprocal_rank = 0.0
        for i, evaluation in enumerate(top_k):
            if evaluation["relevance_score"] >= RELEVANCE_THRESHOLD:
                reciprocal_rank = 1.0 / (i + 1)  # rank = i + 1
                break

        total_reciprocal_rank += reciprocal_rank

    if valid_questions == 0:
        return 0.0

    return total_reciprocal_rank / valid_questions

def main():
    print("=" * 60)
    print("LLM JUDGE RETRIEVAL METRICS (Hit@K, MRR@K, Mean Relevance@K)")
    print("=" * 60)

    data = load_data()
    if data is None:
        return

    print(f"\nTotal questions loaded: {len(data)}")
    print(f"Relevance threshold: >= {RELEVANCE_THRESHOLD}")
    print("-" * 60)

    # Results table
    print("\n{:<6} {:<10} {:<10} {:<14}".format("K", "Hit@K", "MRR@K", "MeanRel@K"))
    print("-" * 60)

    for k in K_VALUES:
        hit_rate, hits, n, _ = calculate_relevance_hit(data, k)
        mrr = calculate_mrr(data, k)
        mean_rel = calculate_mean_relevance(data, k)
        print("{:<6} {:<10.4f} {:<10.4f} {:<14.4f}".format(k, hit_rate, mrr, mean_rel))

    print("\n" + "=" * 60)
    k_target = 15
    failed_ids = get_failed_questions_at_k(data, k_target)
    print(f"\nTarget: Hit@{k_target}")
    print(f"Number of failed questions at Hit@{k_target}: {len(failed_ids)}")
    print("-" * 60)

    if failed_ids:
        print("Failed question IDs:")
        for idx, qid in enumerate(failed_ids, start=1):
            print(f"  {idx}. {qid}")
    else:
        print("No failed questions. All passed the Hit threshold.")

    print("=" * 60)


if __name__ == "__main__":
    main()
