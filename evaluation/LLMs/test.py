import csv
import json
import os
import re
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.avalai.ir/v1"
INPUT_FOLDER = "./evaluation-datasets"

RAW_JSON_OUTPUT = "./model_benchmark_raw.json"
OVERALL_METRICS_CSV = "./benchmark_overall_metrics.csv"
FILE_METRICS_CSV = "./benchmark_file_metrics.csv"

FILE_PATTERN = re.compile(r"^evaluation[_-]dataset[_-]\d+\.json$")

if not API_KEY:
    raise ValueError("کلید OPENAI_API_KEY در فایل .env یافت نشد.")

models: Dict[str, ChatOpenAI] = {
    "Gemini-2.5-Flash": ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
    ),
    "Qwen-3.6-Plus": ChatOpenAI(
        model="qwen3.6-plus",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0
    ),
    "gpt-5.6-sol": ChatOpenAI(
            model="gpt-5.6-sol",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.0
    ),
    "DeepSeek-V3": ChatOpenAI(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0
    )
}

def format_prompt(question: str, options: Any) -> str:
    if isinstance(options, dict):
        options_text = "\n".join(
            [f"{k.upper()}) {v}" for k, v in sorted(options.items())]
        )
    elif isinstance(options, list):
        options_text = "\n".join(
            [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]
        )
    else:
        options_text = str(options)
    return f"Question:\n{question}\n\nOptions:\n{options_text}"


def clean_letter(raw: str) -> str:
    """استخراج گزینه صحیح از خروجی مدل."""
    if not raw or not isinstance(raw, str):
        return "invalid"
    raw = raw.strip()
    if raw == "ERROR":
        return "error"

    if re.fullmatch(r"[a-eA-E]", raw):
        return raw.lower()

    boxed_match = re.findall(r"\\boxed\{([a-eA-E])\}", raw)
    if boxed_match:
        return boxed_match[-1].lower()

    explicit_match = re.findall(
        r"(?:correct\s+(?:option|answer)|final\s+answer|answer\s+is|option|choice)\s*[:\-\s]*\(?([a-eA-E])\)?",
        raw,
        re.IGNORECASE,
    )
    if explicit_match:
        return explicit_match[-1].lower()

    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        match_last = re.findall(r"\b([a-eA-E])\b", last_line)
        if match_last:
            return match_last[-1].lower()

    first_match = re.search(r"\b([a-eA-E])\b", raw)
    if first_match:
        return first_match.group(1).lower()

    return "invalid"


def invoke_model_with_adaptive_retry(
    llm: ChatOpenAI, messages: list, model_name: str, max_retries: int = 5
) -> str:
    """فراخوانی مدل با لاگ وضعیت و مدیریت خطای 429."""
    delay = 3.0
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"ارسال درخواست به {model_name}...", end="", flush=True)
            response = llm.invoke(messages)
            elapsed = time.time() - t0
            print(f" دریافت شد ({elapsed:.1f}s)", flush=True)
            return response.content.strip()
        except Exception as e:
            print(f"خطا!", flush=True)
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                print(
                    f"محدودیت نرخ (429). مکث {delay:.1f} ثانیه... (تلاش {attempt}/{max_retries})",
                    flush=True,
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                if attempt == max_retries:
                    raise e
                print(f"خطای شبکه/ارتباط ({err_str[:80]}). تلاش مجدد پس از 2 ثانیه...", flush=True)
                time.sleep(2.0)
    raise RuntimeError(f"مدل {model_name} پس از {max_retries} تلاش پاسخ نداد.")


def load_existing_results() -> Dict[str, Dict[str, Any]]:
    """بارگذاری و اعتبارسنجی نتایج قبلی."""
    if not os.path.exists(RAW_JSON_OUTPUT):
        return {}

    cleaned_records = {}
    try:
        with open(RAW_JSON_OUTPUT, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            gt = item.get("ground_truth", "")
            for m in list(item.get("raw_responses", {}).keys()):
                raw_resp = item["raw_responses"].get(m, "")
                if raw_resp and raw_resp != "ERROR":
                    parsed = clean_letter(raw_resp)
                    item["predictions"][m] = parsed
                    item["is_correct"][m] = (parsed == gt) and (
                        gt in ["a", "b", "c", "d", "e"]
                    )

            key = f"{item['file']}::{item['id']}"
            cleaned_records[key] = item

        return cleaned_records
    except Exception as e:
        print(f"خطایی در خواندن چک‌پوینت قبلی رخ داد ({e}).", flush=True)
        return {}


def save_checkpoint(results_dict: Dict[str, Dict[str, Any]]):
    """ذخیره امن نتایج روی دیسک."""
    temp_file = RAW_JSON_OUTPUT + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)
    os.replace(temp_file, RAW_JSON_OUTPUT)


def compute_and_save_metrics_csv(results: List[Dict[str, Any]]):
    """محاسبه و ذخیره جدول متریک‌ها."""
    if not results:
        print("داده‌ای برای محاسبه متریک‌ها وجود ندارد.", flush=True)
        return

    all_evaluated_models = set()
    for r in results:
        all_evaluated_models.update(r.get("predictions", {}).keys())
    model_names = sorted(list(all_evaluated_models))

    if not model_names:
        print("هیچ مدلی برای ارزیابی متریک یافت نشد.", flush=True)
        return

    file_list = sorted(list(set(r["file"] for r in results)))
    y_true = [r["ground_truth"] for r in results]
    total_q = len(y_true)

    overall_rows = []
    summary_table_print = []

    for m in model_names:
        y_pred = [r["predictions"].get(m, "invalid") for r in results]
        valid_preds = [1 for p in y_pred if p in ["a", "b", "c", "d", "e"]]
        valid_rate = (len(valid_preds) / total_q) * 100 if total_q > 0 else 0

        acc = accuracy_score(y_true, y_pred) * 100
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        overall_rows.append({
            "Model": m,
            "Accuracy (%)": round(acc, 2),
            "Macro F1": round(f1_macro, 4),
            "Weighted F1": round(f1_weighted, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Valid Rate (%)": round(valid_rate, 2),
            "Total Questions": total_q,
        })
        summary_table_print.append([
            m,
            f"{acc:.2f}%",
            f"{f1_macro:.4f}",
            f"{f1_weighted:.4f}",
            f"{prec_macro:.4f}",
            f"{rec_macro:.4f}",
            f"{valid_rate:.1f}%",
        ])

    with open(OVERALL_METRICS_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=overall_rows[0].keys())
        writer.writeheader()
        writer.writerows(overall_rows)

    file_rows = []
    for fname in file_list:
        f_results = [r for r in results if r["file"] == fname]
        f_true = [r["ground_truth"] for r in f_results]

        row_dict = {"File Name": fname, "Questions Count": len(f_results)}
        for m in model_names:
            f_pred = [r["predictions"].get(m, "invalid") for r in f_results]
            f_acc = accuracy_score(f_true, f_pred) * 100 if f_results else 0
            row_dict[f"{m} Accuracy (%)"] = round(f_acc, 2)

        file_rows.append(row_dict)

    with open(FILE_METRICS_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=file_rows[0].keys())
        writer.writeheader()
        writer.writerows(file_rows)

    print("\nخلاصه متریک‌های کلی:", flush=True)
    headers = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
        "Valid Rate",
    ]
    print(tabulate(summary_table_print, headers=headers, tablefmt="fancy_grid"), flush=True)


def run_benchmark():
    """حلقه اصلی اجرای بنچمارک."""
    print("بررسی ساختار دایرکتوری و فایل‌ها...", flush=True)
    if not os.path.exists(INPUT_FOLDER):
        print(f"پوشه '{INPUT_FOLDER}' یافت نشد.", flush=True)
        return

    json_files = [f for f in sorted(os.listdir(INPUT_FOLDER)) if FILE_PATTERN.match(f)]
    if not json_files:
        print(f"هیچ فایلی با الگوی مشخص‌شده در '{INPUT_FOLDER}' یافت نشد.", flush=True)
        return

    checkpoint_records = load_existing_results()
    if checkpoint_records:
        print(f"تعداد {len(checkpoint_records)} رکورد ذخیره‌شده از قبل لود شد.", flush=True)

    print(f"شروع ارزیابی روی {len(json_files)} فایل با مدل‌های: {list(models.keys())} ...\n", flush=True)

    system_instruction = (
        "You are an expert medical specialist taking a board examination.\n"
        "Read the question and options carefully.\n"
        "Respond ONLY with the single lowercase letter corresponding to the correct option (e.g., 'a', 'b', 'c', 'd', 'e').\n"
        "Do NOT include explanations, punctuation, or any extra text."
    )

    for file_idx, filename in enumerate(json_files, start=1):
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"\n[{file_idx}/{len(json_files)}] باز کردن فایل: {filename}", flush=True)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"خطا در خواندن فایل {filename}: {e}", flush=True)
            continue

        items_list = data if isinstance(data, list) else [data]
        print(f"شامل {len(items_list)} سوال.", flush=True)

        for idx, item in enumerate(items_list, start=1):
            q_id = str(item.get("id", f"{filename}_{idx}"))
            unique_key = f"{filename}::{q_id}"
            ground_truth = clean_letter(str(item.get("correct_answer", "")))

            record = checkpoint_records.get(
                unique_key,
                {
                    "id": q_id,
                    "file": filename,
                    "ground_truth": ground_truth,
                    "predictions": {},
                    "raw_responses": {},
                    "is_correct": {},
                },
            )

            models_to_run = [
                m
                for m in models.keys()
                if m not in record.get("predictions", {})
                or record["predictions"][m] in ["error", "invalid", ""]
            ]

            if not models_to_run:
                continue

            print(f"سوال {idx}/{len(items_list)} (ID: {q_id}) -> نیاز به ارزیابی: {models_to_run}", flush=True)

            question = item.get("question", "")
            options = item.get("options", {})
            prompt = format_prompt(question, options)

            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=prompt),
            ]

            for model_name in models_to_run:
                llm = models[model_name]
                try:
                    raw_text = invoke_model_with_adaptive_retry(llm, messages, model_name, max_retries=5)
                    parsed_letter = clean_letter(raw_text)

                    record["raw_responses"][model_name] = raw_text
                    record["predictions"][model_name] = parsed_letter
                    record["is_correct"][model_name] = (parsed_letter == ground_truth)
                    print(f"پاسخ خام: '{raw_text}' | استخراج‌شده: '{parsed_letter}' | درست؟ {record['is_correct'][model_name]}", flush=True)

                except Exception as e:
                    print(f"خطای نهایی: {e}", flush=True)
                    record["raw_responses"][model_name] = "ERROR"
                    record["predictions"][model_name] = "error"
                    record["is_correct"][model_name] = False

                time.sleep(0.3)

            checkpoint_records[unique_key] = record
            save_checkpoint(checkpoint_records)

    all_final_results = list(checkpoint_records.values())
    print(f"\nپردازش تمام شد. کل سوالات: {len(all_final_results)}", flush=True)
    compute_and_save_metrics_csv(all_final_results)


if __name__ == "__main__":
    run_benchmark()
