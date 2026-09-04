from state import AgenticRAGState
from langgraph.graph import END

def route_question(state: AgenticRAGState) -> str:
    route = str(state.get("route_to", "")).strip().lower()
    lang = str(state.get("detected_language", "")).strip().lower()
    question = str(state.get("user_query", "")).strip()

    print(f"[Router Debug] route_to={route!r}, detected_language={lang!r}, question={question!r}")

    if route in {"out_of_domain", "out-of-domain", "ood"}:
        return "out_of_domain"

    if route in {"urology", "urology_rag", "in_domain", "medical", "medical_urology"}:
        if lang == "fa":
            return "translate_to_en"
        if lang == "en":
            return "retrieve"

        if any('\u0600' <= ch <= '\u06FF' for ch in question):
            print("[Router Debug] Fallback language detection -> fa")
            return "translate_to_en"

        print("[Router Debug] Fallback language detection -> en")
        return "retrieve"

    # 4) Safe fallback
    print("[Router Debug] Unknown route value. Falling back to out_of_domain.")
    return "out_of_domain"


def route_after_rag(state: AgenticRAGState) -> str:
    lang = state["detected_language"]
    if lang == "en":
        return END
    return "translate_to_fa"
