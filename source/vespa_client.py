import requests
import json

VESPA_HOST = "http://localhost:8080"
CONTENT_CLUSTER = "default"  # Vespa cluster ID'yi buraya gir
DOCUMENT_ENDPOINT = f"{VESPA_HOST}/document/v1/{CONTENT_CLUSTER}/documents/docid/"
SEARCH_ENDPOINT = f"{VESPA_HOST}/search/"

from sentence_transformers import SentenceTransformer

class VespaClient:
    def __init__(self):
        self.host = VESPA_HOST
        print(f"VespaClient başlatıldı")
        
    def search(self, query_embedding):
    
        query_data = {
            "yql": "SELECT file_topic FROM documents WHERE ({targetHits: 6} nearestNeighbor(embedding, query_embedding)) OR userQuery() LIMIT 6",
            "input.query(query_embedding)": query_embedding,
            "ranking": "default",
            "hits": 6
        }

        print("En alakalı dosya konusu bulunuyor...")
        response = requests.post(SEARCH_ENDPOINT, json=query_data)

        if response.status_code != 200:
            print("En iyi konu bulunamadı")
            return {"error": response.text}

        search_results = response.json()
        file_topics = [doc["fields"]["file_topic"] for doc in search_results.get("root", {}).get("children", [])]

        if not file_topics:
            print("Soruyla alakalı dosya konusu bulunamadı!")
            return {"error": "No file_topic found"}

        best_file_topic = file_topics[0]
        print(f"En alakalı dosya konusu: {best_file_topic}")
        targetHits = "{targetHits: 9}"


        query_data = {
            "yql": f'SELECT section_topic FROM documents WHERE (({targetHits} nearestNeighbor(embedding, query_embedding)) OR userQuery()) AND file_topic contains "{best_file_topic}" LIMIT 9 ',
            "input.query(query_embedding)": query_embedding,
            "ranking": "default",
            "hits": 9
        }

        print("En alakalı 9 tane bölüm konusu aranıyor...")
        response = requests.post(SEARCH_ENDPOINT, json=query_data)

        if response.status_code != 200:
            print("En alakalı bölüm konusu bulunurken hata meydana geldi!")
            print(response.text)
            return {"error": response.text}

        search_results = response.json()
        section_topics = [doc["fields"]["section_topic"] for doc in search_results.get("root", {}).get("children", [])]

        if not section_topics:
            print("Bölüm konuları bulunamadı!")
            return {"error": "No section_topic found"}


        section_topic_conditions = " OR ".join([f'section_topic CONTAINS "{t}"' for t in section_topics])
        targetHits = "{targetHits: 3}"

        query_data = {
            "yql": f'SELECT section_content FROM documents WHERE (({targetHits} nearestNeighbor(embedding, query_embedding)) OR userQuery()) AND file_topic contains "{best_file_topic}" AND ({section_topic_conditions}) LIMIT 3',
            "input.query(query_embedding)": query_embedding,
            "ranking": "default",
            "hits": 3
        }

        print("En alakalı 3 bölüm içeriği aranıyor...")
        response = requests.post(SEARCH_ENDPOINT, json=query_data)

        if response.status_code != 200:
            print("Bölüm içerikleri aranırken hata meydana geldi!")
            print(response.text)
            return {"error": response.text}

        search_results = response.json()
        section_contents = [doc["fields"]["section_content"] for doc in search_results.get("root", {}).get("children", [])]

        if not section_contents:
            print("Alakalı bölüm içeriği bulunamadı!")
            return {"error": "No section_content found"}

        return {
            "file_topic": best_file_topic,
            "section_topics": section_topics,
            "section_contents": section_contents
        }