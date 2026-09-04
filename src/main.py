from graph import app
from utils import save_log_to_csv


def run_simulations():
    """
    Simulates end-to-end conversations with mock data to test the compiled graph.
    user_role is injected directly (simulating FastAPI endpoints).
    """
    test_cases = [
        (
            "Test 1: Persian Doctor (Clinical/Academic terms in Farsi)", 
            "پروتکل درمانی استاندارد برای کارسینومای یوروتلیال غیر تهاجمی مثانه (NMIBC) در بیماران High-risk چیست؟",
            "doctor"
        ),
        (
            "Test 2: English Patient (Layman terms)", 
            "It hurts when I pee and I see blood in my urine, should I be worried?",
            "patient"
        ),
        (
            "Test 3: Mixed Language Doctor (Finglish/Mixed Clinical)", 
            "برای بیماری که با BPH مراجعه کرده و سطح PSA بالا داره، آیا Multiparametric MRI قبل از بیوپسی recommend میشه؟",
            "doctor"
        ),
        (
            "Test 4: Out of Domain (English)", 
            "What is the formula for calculating the kinetic energy of an electron?",
            "patient"  # در Out of domain نقش کاربر اهمیتی ندارد و گراف فوراً قطع می‌شود
        ),
        (
            "Test 5: Urodynamic Specific Doctor Context", 
            "در تفسیر نمودارهای یورودینامیک، مشاهده Detrusor Overactivity همراه با فشار بالای نشت (Leak Point Pressure) معمولاً نشانه چیست؟",
            "doctor"
        ),
        (
            "Test 6: Persian Patient (Same query as Test 1 to compare tone)", 
            "پروتکل درمانی استاندارد برای سرطان مثانه در بیماران پرخطر چیست؟",
            "patient"
        )
    ]

    for title, question, role in test_cases:
        print(f"\n=== {title} ===")
        print(f"[Input Role]: {role}")
        print(f"[Query]: {question}")
        
        inputs = {
            "user_query": question,
            "user_role": role,
        }
        
        # اجرای گراف
        res = app.invoke(inputs)
        
        domain = res.get("route_to", "N/A")
        language = res.get("detected_language", "N/A")
        user_type = res.get("user_role", role)
        generation_output = res.get("final_output", "N/A")

        print(f"\n[Domain]: {domain} | [Language]: {language} | [Role]: {user_type}")
        print(f"\n[Final Output]:\n{generation_output}")
        print("\n" + "="*50 + "\n")
        
        save_log_to_csv(question, domain, language, user_type, generation_output)

if __name__ == "__main__":
    run_simulations()
