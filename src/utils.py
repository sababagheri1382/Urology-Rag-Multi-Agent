import os
import csv

def save_log_to_csv(question: str, domain: str, language: str, user_type: str, generation: str, filename: str = "rag_logs.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Question", "Domain", "Language", "User Type", "Generation Output"])
        
        writer.writerow([question, domain, language, user_type, generation])
