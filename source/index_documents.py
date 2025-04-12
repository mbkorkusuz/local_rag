import os
import json
import requests
from sentence_transformers import SentenceTransformer
import hashlib  # To generate unique document IDs
import time  # To add delays if needed

# Vespa Configuration
VESPA_NAMESPACE = "default"
VESPA_DOCUMENT_TYPE = "documents"
VESPA_ENDPOINT = f"http://localhost:8080/document/v1/{VESPA_NAMESPACE}/{VESPA_DOCUMENT_TYPE}/docid"

# Load Sentence Transformer for Embeddings
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

DATA_DIR = "../MEB_data"  # Directory containing text files

def parse_txt_file(file_path):
    print(f"İşlenen dosya: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    file_topic = None
    sections = []
    current_section_topic = None
    current_section_content = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Extract file topic
        if line.startswith(">>>") and i + 1 < len(lines) and lines[i + 1].startswith("Konu:"):
            file_topic = lines[i + 1].replace("Konu:", "").strip()
            print(f"İşlenen dosya konusu: {file_topic}")
            i += 2  # Skip next line since it's already processed
            continue

        if line.startswith("#SECTION_START#"):
            if current_section_topic and current_section_content:
                sections.append({
                    "file_topic": file_topic,
                    "section_topic": current_section_topic,
                    "section_content": "\n".join(current_section_content)
                })
            current_section_topic = None
            current_section_content = []
            i += 1
            continue

        if line.startswith("#SECTION_TOPIC:"):
            current_section_topic = line.replace("#SECTION_TOPIC:", "").strip("# ")
            i += 1
            continue

        if line.startswith("#SECTION_END#"):
            if current_section_topic and current_section_content:
                sections.append({
                    "file_topic": file_topic,
                    "section_topic": current_section_topic,
                    "section_content": "\n".join(current_section_content)
                })
            current_section_content = []
            current_section_topic = None
            i += 1
            continue

        if current_section_topic:
            current_section_content.append(line)

        i += 1

    if current_section_topic and current_section_content:
        sections.append({
            "file_topic": file_topic,
            "section_topic": current_section_topic,
            "section_content": "\n".join(current_section_content)
        })

    return sections

def generate_embeddings(text):
    return model.encode(text).tolist()

def create_unique_document_id(file_topic, section_topic):
    """ Generate a truly unique document ID for each section """
    return f"{hashlib.md5(f'{file_topic}_{section_topic}'.encode()).hexdigest()}"

def index_documents():
    if not os.path.exists(DATA_DIR):
        return

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)

        if not file_name.endswith(".txt"):
            continue
        sections = parse_txt_file(file_path)

        for section in sections:
            #if not section["file_topic"] or not section["section_topic"]:
            #    continue
            section["embedding"] = model.encode(section["section_content"]).tolist()
            #section["embedding"] = generate_embeddings(section["section_content"])
            
            # Generate unique document ID
            document_id = create_unique_document_id(section["file_topic"], section["section_topic"])

            vespa_doc = {
                "fields": {
                    "file_topic": section["file_topic"],
                    "section_topic": section["section_topic"],
                    "section_content": section["section_content"],
                    "embedding": section["embedding"]
                }
            }

            # Print the document being sent to Vespa (for debugging)
            #print(f"{document_id} Vespaya gönderiliyor. Vespa: {json.dumps(vespa_doc, indent=2)}")

            # Send to Vespa and check response
            response = requests.post(f"{VESPA_ENDPOINT}/{document_id}", json=vespa_doc)


            if response.status_code != 200:
                print(f"{document_id} gönderimi başarısız - Vespa Cevabı: {response.text}")
            else:
                print(f"{document_id} başarıyla gönderildi")

            # Add a short delay to prevent request conflicts
            time.sleep(0.5)

if __name__ == "__main__":
    index_documents()
