import os
import re
import json
import uuid
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

def clean_text(text: str) -> str:
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'([a-zA-Z]+)-\n([a-zA-Z]+)', r'\1\2\n', text)
    return text

def process_document(input_filepath: Path, output_filepath: Path, doc_id: str):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"error reading file {input_filepath.name}: {e}")
        return

    cleaned_text = clean_text(raw_text)

    headers_to_split_on = [
        ("#######", "Chapter")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )

    md_header_splits = markdown_splitter.split_text(cleaned_text)

    structured_data = []
    
    for index, doc in enumerate(md_header_splits):
        text_content = doc.page_content.strip()
        
        if not text_content:
            continue
            
        extracted_metadata = doc.metadata
        chapter_name = extracted_metadata.get("Chapter", "بدون عنوان")
        
        char_count = len(text_content)
        token_count = len(text_content.split())
        
        chunk_data = {
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": index,
            "doc_id": doc_id,
            "source_file": input_filepath.name,
            "chapter": chapter_name,
            "breadcrumb": f"{doc_id} > {chapter_name}",
            "text": text_content,
            "metadata": {
                "char_count": char_count,
                "token_count": token_count
            }
        }
        structured_data.append(chunk_data)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)
        
    print(f"processing of file {input_filepath.name} done. {len(structured_data)}  number of chunks saved in {output_filepath.name}")

def batch_process_folder(input_dir_path: str, output_dir_path: str):
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"no .txt files in'{input_dir.resolve()}'")
        return

    print(f"found {len(txt_files)} number of files. starting...\n" + "="*60)

    for txt_file in txt_files:
        file_stem = txt_file.stem 
        output_file = output_dir / f"{file_stem}.json"
        
        process_document(
            input_filepath=txt_file,
            output_filepath=output_file,
            doc_id=file_stem
        )

    print("="*60 + "\n done")

if __name__ == "__main__":
    INPUT_DIRECTORY = "../../raw-texts"        
    OUTPUT_DIRECTORY = "processed-jsons"   
    
    batch_process_folder(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
