from graph import app
from utils import save_log_to_csv

def generate_answer(question: str, user_type: str) -> str:
    """
    Executes the LangGraph compiled medical RAG pipeline
    and logs the interaction.
    """

    inputs ={
        "user_query": question,
        "user_role": user_type,
    }

    result = app.invoke(inputs)

    domain = result.get("route_to", "N/A")
    language = result.get("detected_language", "N/A")
    user_role = result.get("user_role", user_type)
    generation_output = result.get("final_output", "پاسخی دریافت نشد.")
    
    try:
        save_log_to_csv(question, domain, language, user_role, generation_output)
    except Exception as e:
        print(f"[Warning] Failed to save log to CSV: {e}")
        
    return generation_output
    