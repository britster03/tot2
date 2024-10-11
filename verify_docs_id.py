import json

def verify_unique_doc_ids(corpus_path):
    doc_ids = set()
    duplicates = False
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line)
                doc_id = doc.get('doc_id')
                if doc_id in doc_ids:
                    print(f"Duplicate doc_id found: {doc_id} at line {line_num}")
                    duplicates = True
                doc_ids.add(doc_id)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed at line {line_num}: {e}")
                duplicates = True
    if not duplicates:
        print("All doc_ids are unique.")
    else:
        print("Duplicates found in doc_ids. Please ensure all doc_ids are unique.")

if __name__ == "__main__":
    corpus_path = 'data/corpus.jsonl'  # Update if necessary
    verify_unique_doc_ids(corpus_path)
