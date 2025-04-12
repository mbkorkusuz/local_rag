from fastapi import FastAPI
from pydantic import BaseModel
from vespa_client import VespaClient 
from main import main as langgraph_workflow  # import LangGraph workflow
from sentence_transformers import SentenceTransformer
app = FastAPI()
vespa = VespaClient()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    user_query = request.query

    # Retrieve relevant sections using Vespa
    query_embedding = model.encode(user_query).tolist()
    vespa_response = vespa.search(query_embedding)
    
    # Extract fields
    section_topics = vespa_response.get("section_topics", [])
    section_contents = vespa_response.get("section_contents", [])

    
    retrieved_docs = [
        {"section_topic": topic, "section_content": content}
        for topic, content in zip(section_topics, section_contents)
    ]

    #print(section_contents)

    input_data = {
        "question": user_query,
        "question_transformed_for_relevance": 0,
        "question_transformed_for_regenerate": 0,
        "generation_count": 0,
        "documents": section_contents,
        "generation": ""
    }
    
    
    app_instance = langgraph_workflow()
    #app_instance = workflow.compile()
    output = app_instance.invoke(input_data)
    llm_response = output.get("generation", "No response generated.")

    return {
        "retrieved_documents": retrieved_docs,
        "llm_response": llm_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)