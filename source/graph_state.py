import os
import re
from data_models_and_functions import retrieve_best_match, generate_answer, fact_grader, answer_grader, question_rewriter, relevance_checker, dont_know_answer, lack_of_documents
import streamlit as st

## Nodes

def retrieve(state):
    """Retrieve documents

       Args: 
           state (dict): The current graph state

       Returns: 
           state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVING---")

    question = state["question"]
    documents = retrieve_best_match(query=question)
    state["documents"] = documents

    return state


def generate(state):
    """Generate answer

       Args:
           state (dict): The current graph state

       Returns:
           state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"] 

    # RAG Generation
    document = "\n".join(documents)
    #for section in documents:
        #document = document + section + "\n"

    generation = generate_answer(question=question, document=document)

    state["generation"] = generation

    return state

def grade_documents(state):
    """Determines whether the retrieved documents are related or not
    
       Args:
           state (dict): The current graph state

       Returns:
           state (dict): Updates documents key with only filtered relevant documents
    """

    
    print("---CHECK RELEVANCE OF DOCUMENTS---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    
    for i in range(len(documents)):
        document = documents[i]
        output = relevance_checker(question=question, document=document)

        if "yes" in output:
            print("---DOCUMENT IS RELEVANT---")
            filtered_docs.append(output)
        else:
            print("---DOCUMENT IS IRRELEVANT---")

            
    
    state["documents"] = filtered_docs

    return state

def transform_query_for_relevance(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    
    print("---TRANSFORM QUERY---")
    question = state["question"]
    question_transformed_for_relevance = state["question_transformed_for_relevance"]
    question_transformed_for_relevance += 1

    better_question = question_rewriter(question=question)
    state["question"] = better_question
    state["question_transformed_for_relevance"] = question_transformed_for_relevance

    return state

def transform_query_for_regenerate(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    
    print("---TRANSFORM QUERY---")

    question = state["question"]
    question_transformed_for_regenerate = state["question_transformed_for_regenerate"]
    question_transformed_for_regenerate += 1

    better_question = question_rewriter(question=question)

    state["question"] = better_question
    state["question_transformed_for_regenerate"] = question_transformed_for_regenerate

    return state

def dont_know(state):
    print("---DONT KNOW---")
    question = state["question"]
    sorry_text = dont_know_answer(question=question)

    state["generation"] = sorry_text

    return state

def lack_documents(state):
    print("---NEED FOR MORE RELEVANT DOCUMENTS---")
    question = state["question"]
    lack_text = lack_of_documents(question=question)

    state["generation"] = lack_text

    return state


## Edges
def should_transform_for_relevance(state):
    """
        Look for transform count and decide whether transform or not.
    """
    should_transform_for_relevance_count = state["question_transformed_for_relevance"]

    if should_transform_for_relevance_count <= 1:
        return "transform"
    elif should_transform_for_relevance_count > 1:
        print("---EXCEED TRANSFORMING QUERY COUNT FOR RELEVANCE. DECISION: END THE PROCESS---")
        return "stop"

def should_transform_for_regenerate(state):
    """
        Look for transform count and decide whether transform or not.
    """
    should_transform_for_regenerate_count = state["question_transformed_for_regenerate"]

    if should_transform_for_regenerate_count <= 1:
        return "transform"
    elif should_transform_for_regenerate_count > 1:
        print("---EXCEED TRANSFORMING QUERY COUNT FOR REGENERATE. DECISION: END THE PROCESS---")
        return "stop"


def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a question.

       Args:
           state (dict): The current graph state

       Returns:
           str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_docs = state["documents"]

    if len(filtered_docs) <= 1:
        print("---DECISION: NOT ENOUGH DOCUMENTS ARE RELEVANT, TRANSFORMING QUERY---")
        return "transform_query"

    else:
        print("---DECISION: ENOUGH RELEVANT DOCUMENTS, GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded by retrieved docs and answers the question
    
       Args:
           state (dict): The current graph state

       Returns:
           str: Decision for next node call
    """
    print("---CHECK IF GENERATION IS GROUNDED BY DOCS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generation_count = state["generation_count"]
    generation_count += 1
    state["generation_count"] = generation_count
    
    document = "\n".join(documents)
    #for section in documents:
    #    document = document + section + "\n"
    
    fact = fact_grader(document=document, generation=generation) # Hallucination checker

    if fact == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        print("---GRADE GENERATION vs QUESTION---")
        answer_grade = answer_grader(question, generation)
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES THE QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not useful"

    else:
        if generation_count >= 2:
            print("---DECISION: STOP, GENERATION COUNT IS EXCEED")
            return "stop"
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"    