import numpy as np
import contextlib
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import faiss
from sentence_transformers import SentenceTransformer
from vespa_client import VespaClient

vespa = VespaClient()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

generate_prompt_template = """
Below, you will see a question and a document example. Answer the question using only the information from the document.\n

Question:\n
{user_question}

Document:\n
{document_text}
\n
Answer:
"""

fact_prompt_template = """
Below, you will see a document and an LLM output. Check whether the given document and the LLM output are consistent with each other.\n
Answer only with 'yes' or 'no'. 'Yes' should indicate that they are consistent.\n
\n
Document:\n
{document_text}

LLM Output:\n
{generation}
\n
Answer:
"""

answer_prompt_template = """
Below, you will see a question and an LLM output. Check whether the LLM output answers the question.\n
Answer only with 'yes' or 'no'. 'Yes' should indicate that the answer addresses the question.\n
\n

Question:\n
{user_question}

LLM Output:\n
{generation}
\n
Answer:
"""

rewrite_prompt_template = """
You are an assistant that rewrites the given question.\n
Rewrite the question by optimizing it.\n
Do not change the intended meaning.\n
Do not provide any explanations. Only output the revised question.\n

Question:\n
{user_question}
\n
Revised Version:
"""

relevance_prompt_template = """
Below, you will see a question and a document example. Check whether question and the document is related or not. Be strict.\n
Answer only with 'yes' or 'no'. 'Yes' should indicate that there is a connection. Do not explain anything. Just answer 'yes' or 'no'\n

Question:\n
{user_question}

Document:\n
{document_text}
\n
Answer:
"""

dont_know_answer_template = """
Below you will see a question. Write a response that tells the user you don't know the answer.

Question:\n
{user_question}

"""

lack_of_documents_template = """
Below you will see a question. Write a response that tells the user you need more relevant documents for an answer.

Question:
{user_question}
"""

generate_prompt = PromptTemplate.from_template(generate_prompt_template)
fact_prompt = PromptTemplate.from_template(fact_prompt_template)
answer_prompt = PromptTemplate.from_template(answer_prompt_template)
rewrite_prompt = PromptTemplate.from_template(rewrite_prompt_template)
relevance_prompt = PromptTemplate.from_template(relevance_prompt_template)
dont_know_answer_prompt = PromptTemplate.from_template(dont_know_answer_template)
lack_of_documents_prompt = PromptTemplate.from_template(lack_of_documents_template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm_for_yes_no = LlamaCpp(
    model_path="mistral-7b-instruct-v0.2-turkish.Q5_K_M.gguf",
    temperature=0.1,
    max_tokens=10, 
    top_k = 100,
    top_p=0.1, ## nucleus sampling. 
    callback_manager=callback_manager,
    n_ctx = 4096,
    verbose=False,  # Verbose is required to pass to the callback manager
)
llm = LlamaCpp(
    model_path="mistral-7b-instruct-v0.2-turkish.Q5_K_M.gguf",
    temperature=1.3,
    max_tokens=300,
    top_p=0.9, ## nucleus sampling. 
    callback_manager=callback_manager,
    n_ctx = 4096,
    verbose=False,  # Verbose is required to pass to the callback manager
)
def retrieve_best_match(query):
    """ Retrieves the most relevant documents from Vespa """
    print(f"Kullanıcı sorusu için dökümanlar aranıyor: {query}")
    
    query_embedding = model.encode(query).tolist()
    
    # Vespa search query
    results = vespa.search(query_embedding)
    
    documents = results.get("section_contents", [])
    
    print(f"{len(documents)} tane döküman Vespadan getirildi.")
    return documents


def generate_answer(question, document):
    formatted_prompt = generate_prompt.format(user_question=question, document_text=document)

    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm.invoke(formatted_prompt)
        
    return output

def fact_grader(document, generation):
    formatted_prompt = fact_prompt.format(document_text=document, generation=generation)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm_for_yes_no.invoke(formatted_prompt).lower()

    if "yes" in output:
        return "yes"
    elif "no" in output:
        return "no"
    
def answer_grader(question, generation):
    formatted_prompt = answer_prompt.format(user_question=question, generation=generation)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm_for_yes_no.invoke(formatted_prompt).lower()

    if "yes" in output:
        return "yes"
    elif "no" in output:
        return "no"
    
def question_rewriter(question):
    formatted_prompt = rewrite_prompt.format(user_question=question)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm.invoke(formatted_prompt)
    return output

def relevance_checker(question, document):
    formatted_prompt = relevance_prompt.format(user_question=question, document_text=document)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm_for_yes_no.invoke(formatted_prompt).lower()
    return output

def dont_know_answer(question):
    formatted_prompt = dont_know_answer_prompt.format(user_question = question)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm.invoke(formatted_prompt)
    return output

def lack_of_documents(question):
    formatted_prompt = lack_of_documents_prompt.format(user_question = question)
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = llm.invoke(formatted_prompt)
    return output