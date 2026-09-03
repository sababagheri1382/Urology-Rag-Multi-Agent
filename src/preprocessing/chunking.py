import json
import uuid
import os
import glob
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 150

input_dir = "../processed-jsons"
output_json_path = "final_granular_chunks_all.json"

# Table handling
def is_table_row(line, min_pipes=2):
    return line.count("|") >= min_pipes


def is_caption_line(line):
    normalized = re.sub(r"^\s*#+\s*", "", line.strip())
    pattern = re.compile(r"\b(?:e)?table\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
    return bool(pattern.search(normalized))


def extract_tables_with_captions(text, min_rows=2, max_caption_lookback=3):
    lines = text.split("\n")
    n = len(lines)
    blocks = []
    used = set()

    i = 0
    while i < n:
        if i in used:
            i += 1
            continue

        if is_table_row(lines[i]):
            start = i
            end = i

            while end + 1 < n and is_table_row(lines[end + 1]):
                end += 1

            if (end - start + 1) >= min_rows:
                caption_start = start

                for lookback in range(1, max_caption_lookback + 1):
                    prev = start - lookback
                    if prev < 0:
                        break

                    prev_line = lines[prev].strip()

                    if is_table_row(prev_line):
                        break
                    if not prev_line:
                        continue
                    if is_caption_line(prev_line):
                        caption_start = prev
                        break
                    else:
                        break

                block = "\n".join(lines[caption_start:end + 1])
                blocks.append(block)

                for idx in range(caption_start, end + 1):
                    used.add(idx)

                i = end + 1
                continue

        i += 1

    return blocks


def protect_tables(text):
    table_map = {}
    blocks = extract_tables_with_captions(text)

    for i, block in enumerate(blocks):
        key = f"@@TABLE_{i}_{uuid.uuid4().hex}@@"
        if block in text:
            text = text.replace(block, key, 1)
            table_map[key] = block

    return text, table_map


def restore_tables(text, table_map):
    for key, block in table_map.items():
        text = text.replace(key, block)
    return text


def contains_table(text):
    table_like = sum(
        1 for line in text.splitlines() if is_table_row(line, min_pipes=2)
    )
    return table_like >= 2


def clean_chunk_text(text):
    return text.strip()


def is_heading_only(text):
    t = text.strip()
    if "\n" not in t and re.match(r"^\s*#{1,}\s+\S+", t):
        return True

    return False


def is_too_short_or_weak(text, min_len=MIN_CHUNK_LENGTH):
    t = text.strip()
    if not t:
        return True
    if is_heading_only(t):
        return True
    if len(t) < min_len:
        return True
    return False


def main():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []
    global_chunk_index = 0
    total_input_items = 0

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to process...")

    for file_path in json_files:
        print(f"Processing: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as f:
            structural_data = json.load(f)

        total_input_items += len(structural_data)

        for item in structural_data:
            parent_text = item.get("text", "")
            parent_id = item.get("chunk_id", "")
            doc_id = item.get("doc_id", "")
            breadcrumb = item.get("breadcrumb", "")

            if not parent_text or not parent_text.strip():
                continue

            protected_text, table_map = protect_tables(parent_text)
            splits = text_splitter.split_text(protected_text)

            pending_prefix = ""   

            for split_text in splits:
                split_text = restore_tables(split_text, table_map)
                split_text = clean_chunk_text(split_text)

                if not split_text:
                    continue

                if pending_prefix:
                    split_text = pending_prefix + "\n\n" + split_text
                    pending_prefix = ""

                split_has_table = contains_table(split_text)

                if is_too_short_or_weak(split_text):
                    if pending_prefix:
                        pending_prefix += "\n\n" + split_text
                    else:
                        pending_prefix = split_text
                    continue

                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "parent_chunk_id": parent_id,
                    "chunk_index": global_chunk_index,
                    "doc_id": doc_id,
                    "breadcrumb": breadcrumb,
                    "text": split_text,
                    "has_table": split_has_table
                }
                final_chunks.append(chunk)
                global_chunk_index += 1

            if pending_prefix:
                if final_chunks and final_chunks[-1]["parent_chunk_id"] == parent_id:
                    final_chunks[-1]["text"] += "\n\n" + pending_prefix
                    final_chunks[-1]["has_table"] = (
                        final_chunks[-1]["has_table"] or contains_table(pending_prefix)
                    )
                else:
                    if not is_heading_only(pending_prefix):
                        final_chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "parent_chunk_id": parent_id,
                            "chunk_index": global_chunk_index,
                            "doc_id": doc_id,
                            "breadcrumb": breadcrumb,
                            "text": pending_prefix,
                            "has_table": contains_table(pending_prefix)
                        })
                        global_chunk_index += 1

    print(f"Saving all chunks to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)

    print("\nChunking completed successfully!")
    print(f"Total Files Processed: {len(json_files)}")
    print(f"Total Structural Chunks (Input): {total_input_items}")
    print(f"Total Granular Chunks (Output): {len(final_chunks)}")
    print(f"Chunks with tables: {sum(1 for c in final_chunks if c['has_table'])}")


if __name__ == "__main__":
    main()
