import streamlit as st
import requests

st.title("MEBChat")
st.write("MEB dökümanlarıyla güçlendirilmiş LLM chatbotunuza sorularınızı sormaya başlayın!")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Sorunuz: ")
if query:
    # append user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # send request
    response = requests.post("http://localhost:8000/query", json={"query": query})
    if response.status_code == 200:
        data = response.json()
        retrieved_docs = data.get("retrieved_documents", [])
        llm_response = data.get("llm_response", "No response generated.")


        doc_summaries = "\n\n".join([f"**Madde Konusu:\n\n{doc.get('section_topic', 'No Title')}**\n\n**Madde İçeriği:\n\n{doc.get('section_content', 'No summary available.')}" for doc in retrieved_docs])
        final_response = f"{llm_response}\n\n---\n**Kaynak Paragraflar:**\n\n\n{doc_summaries}"
        
        # append llm response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant"):
            st.markdown(final_response)
    else:
        st.error("Error connecting to the backend.")
