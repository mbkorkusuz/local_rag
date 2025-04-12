from langgraph.graph import START, END, StateGraph

from graph_state import retrieve, grade_documents, generate, transform_query_for_relevance, transform_query_for_regenerate, decide_to_generate,grade_generation_v_documents_and_question, should_transform_for_relevance, should_transform_for_regenerate, dont_know, lack_documents

from typing import List
from typing_extensions import TypedDict

## Graph data model
class GraphState(TypedDict):
    """Represents the state of the graph

       Attributes:
           question: question
           generation: LLM generation
           documents: retrieved documents
           question_transformed_for_relevance: count of transformation of query for relevance check
           question_transformed_for_regenerate: count of transformation of query for generating check
           generation_count: count of how many times LLM generated an answer
    """
    question: str
    generation: str
    documents: List[str]
    question_transformed_for_relevance: int
    question_transformed_for_regenerate: int
    generation_count: int

def test(app, question):
    output = app.invoke(question)
    print(output["generation"])


def main():
    workflow = StateGraph(GraphState)

    ## Defining the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query_for_relevance", transform_query_for_relevance)
    workflow.add_node("transform_query_for_regenerate", transform_query_for_regenerate)
    workflow.add_node("dont_know", dont_know)
    workflow.add_node("lack_documents", lack_documents)


    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges("grade_documents", 
                                   decide_to_generate, 
                                   {"transform_query": "transform_query_for_relevance", "generate": "generate"})
    workflow.add_conditional_edges("transform_query_for_relevance", 
                                   should_transform_for_relevance, 
                                   {"transform": "retrieve", "stop": "lack_documents"}) ## dont know from lack of documents
    workflow.add_conditional_edges("generate", 
                                   grade_generation_v_documents_and_question, 
                                   {"not supported": "generate", "useful": END, "not useful": "transform_query_for_regenerate", "stop": "dont_know"})
    workflow.add_conditional_edges("transform_query_for_regenerate", 
                                   should_transform_for_regenerate, 
                                   {"transform": "retrieve", "stop": "dont_know"}) ## dont know
    
    workflow.add_edge("dont_know", END)
    workflow.add_edge("lack_documents", END)

    app = workflow.compile()

    return app

    #question = "Öğretmenlerin nitelikleri ve seçimi neye göre yapılır?"

    #input = {"question": question, "question_transformed_for_relevance": 0, "question_transformed_for_regenerate": 0, "generation_count": 0}

    #test(app = app, question=input)


if __name__ == "__main__":
    main()