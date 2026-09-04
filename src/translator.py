import os
from dotenv import load_dotenv
import sys

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import translator_llm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Persian display in terminal
import arabic_reshaper
from bidi.algorithm import get_display


def print_fa(text):
    """
    Fix Persian/Arabic text ONLY for terminal display.
    The original text remains unchanged.
    """
    if not isinstance(text, str):
        text = str(text)

    reshaped = arabic_reshaper.reshape(text)
    display_text = get_display(reshaped)

    print(display_text)


from state import AgenticRAGState

fa_to_en_system_prompt = """
You are a highly skilled medical translator and query retrieval expert.
Translate the user's Persian medical question into formal English.

Strict Rules:
- DO NOT add introductory or concluding remarks.
- DO NOT answer the question.
- CRITICAL: Transliterate drug names exactly.
- QUERY OPTIMIZATION: If the query is technical/clinical, append the most relevant standard medical 
  keywords, therapies, or staging terms (in English) to the translated query to optimize 
  vector database retrieval.

Examples:
Input: نشانه های سرطان معده چیست؟
Output: What are the symptoms of stomach cancer?

Input: پروتکل درمانی NMIBC پرخطر چیست؟
Output: What is the standard treatment protocol for high-risk non-muscle-invasive bladder cancer (NMIBC)? Include keywords: TURBT, intravesical BCG, risk stratification, induction, maintenance.
"""

Farsi_to_English_prompt = ChatPromptTemplate.from_messages([
    ("system", fa_to_en_system_prompt),
    (
        "human",
        """Translate the following Persian text into English, including optimization keywords if necessary.

Persian text:
{text}

English translation:"""
    ),
])


en_to_fa_system_prompt = """
You are an expert medical translator.
Translate the English medical text into fluent, formal Persian (Farsi).

Strict Rules:
- DO NOT add introductory or concluding remarks.
- CRITICAL: Use standard, real Persian words. DO NOT invent or merge words.
- ONLY output the translated Persian text.
"""

English_to_Farsi_prompt = ChatPromptTemplate.from_messages([
    ("system", en_to_fa_system_prompt),
    (
        "human",
        """Translate the following English text into Persian.

English text:
{text}

Persian translation:"""
    ),
])



fa_to_en_chain = (
    Farsi_to_English_prompt
    | translator_llm
    | StrOutputParser()
)

en_to_fa_chain = (
    English_to_Farsi_prompt
    | translator_llm
    | StrOutputParser()
)


def translate_to_english_node(state: AgenticRAGState) -> AgenticRAGState:

    persian_query = state["user_query"]

    english_translation = (
        fa_to_en_chain.invoke({
            "text": persian_query
        }).strip()
    )

    return {
        "english_query": english_translation
    }


def translate_to_persian_node(state: AgenticRAGState) -> AgenticRAGState:

    english_response = state["rag_response"]

    persian_translation = (
        en_to_fa_chain.invoke({
            "text": english_response
        }).strip()
    )

    return {
        "final_output": persian_translation
    }


# تست
if __name__ == "__main__":
    import time
    import traceback
    from datetime import datetime

    output_file = os.path.join(
        current_dir,
        "translation_test_results.txt"
    )

    test_cases_fa = [
        "عوارض جانبی داروی تامسولوسین در درمان بزرگی خوش‌خیم پروستات چیست؟",
        "علائم سرطان مثانه چیست؟",
        "آیا عفونت ادراری می‌تواند باعث خون در ادرار شود؟",
        "درمان بزرگی خوش‌خیم پروستات چیست؟",
        "تفاوت بین بی‌اختیاری ادرار استرسی و فوریتی چیست؟",
    ]

    test_cases_en = [
        "Tamsulosin is commonly used to treat benign prostatic hyperplasia.",
        "Common side effects include dizziness, headache, and retrograde ejaculation.",
        "The patient presents with urinary frequency, urgency, and nocturia.",
        "Benign prostatic hyperplasia is a non-cancerous enlargement of the prostate gland.",
        "Hematuria may be caused by urinary tract infections, kidney stones, or bladder tumors.",
    ]

    drug_tests = [
        "تامسولوسین",
        "فیناستراید",
        "دوتاستراید",
        "سیپروفلوکساسین",
        "تادالافیل",
        "اکسی‌بوتینین",
    ]

    def write_header(file, title):
        header = "\n" + "=" * 70 + "\n"
        header += f"TEST: {title}\n"
        header += "=" * 70 + "\n"

        print(header, flush=True)
        file.write(header)
        file.flush()

    def run_test(file, chain, text, direction, test_number):
        print("\n" + "-" * 70, flush=True)
        print(
            f"[{direction}][Test {test_number}] "
            f"Started at {datetime.now().strftime('%H:%M:%S')}",
            flush=True,
        )
        print(
            f"[{direction}][Test {test_number}] Sending request to LLM...",
            flush=True,
        )

        started_at = time.monotonic()

        try:
            # هر تست فقط یک بار به API ارسال می‌شود
            result = chain.invoke({
                "text": text
            })

            elapsed = time.monotonic() - started_at

            if result is None:
                raise ValueError("The LLM returned None.")

            result = str(result).strip()

            print(
                f"[{direction}][Test {test_number}] "
                f"LLM responded in {elapsed:.2f} seconds.",
                flush=True,
            )
            print(
                f"[{direction}][Test {test_number}] "
                f"Response length: {len(result)} characters.",
                flush=True,
            )

            print("\nInput:", flush=True)
            print(text, flush=True)

            print("\nOutput:", flush=True)

            # فعلاً print معمولی تا bidi در تست دخالت نکند
            print(result, flush=True)

            file.write(f"\n--- {direction} Test {test_number} ---\n")
            file.write(f"Elapsed: {elapsed:.2f} seconds\n")
            file.write("Status: SUCCESS\n")
            file.write("Input:\n")
            file.write(text + "\n")
            file.write("\nOutput:\n")
            file.write(result + "\n")
            file.flush()

            return True

        except Exception as exc:
            elapsed = time.monotonic() - started_at
            error_traceback = traceback.format_exc()

            print(
                f"[{direction}][Test {test_number}] "
                f"FAILED after {elapsed:.2f} seconds.",
                flush=True,
            )
            print(
                f"Error type: {type(exc).__name__}",
                flush=True,
            )
            print(
                f"Error message: {exc}",
                flush=True,
            )
            print(error_traceback, flush=True)

            file.write(f"\n--- {direction} Test {test_number} ---\n")
            file.write(f"Elapsed: {elapsed:.2f} seconds\n")
            file.write("Status: FAILED\n")
            file.write(f"Error type: {type(exc).__name__}\n")
            file.write(f"Error message: {exc}\n")
            file.write("Traceback:\n")
            file.write(error_traceback + "\n")
            file.flush()

            return False

        finally:
            # کاهش احتمال rate limit
            time.sleep(1)

    successful_tests = 0
    failed_tests = 0

    with open(output_file, "w", encoding="utf-8") as f:
        write_header(f, "Persian -> English")

        for i, text in enumerate(test_cases_fa, 1):
            success = run_test(
                file=f,
                chain=fa_to_en_chain,
                text=text,
                direction="FA->EN",
                test_number=i,
            )

            if success:
                successful_tests += 1
            else:
                failed_tests += 1

        write_header(f, "English -> Persian")

        for i, text in enumerate(test_cases_en, 1):
            success = run_test(
                file=f,
                chain=en_to_fa_chain,
                text=text,
                direction="EN->FA",
                test_number=i,
            )

            if success:
                successful_tests += 1
            else:
                failed_tests += 1

        write_header(f, "Drug Names")

        for i, drug in enumerate(drug_tests, 1):
            question = f"عوارض جانبی داروی {drug} چیست؟"

            success = run_test(
                file=f,
                chain=fa_to_en_chain,
                text=question,
                direction="DRUG FA->EN",
                test_number=i,
            )

            if success:
                successful_tests += 1
            else:
                failed_tests += 1

    print("\n" + "=" * 70, flush=True)
    print("All tests completed.", flush=True)
    print(f"Successful tests: {successful_tests}", flush=True)
    print(f"Failed tests: {failed_tests}", flush=True)
    print(f"Results saved to: {output_file}", flush=True)
    print("=" * 70, flush=True)
