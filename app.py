import os

import docx2txt
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import streamlit as st
from streamlit_chat import message

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()


@st.cache_data
def load_into_chroma(docs):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(text_splitter)
    global db_chroma
    db_chroma = Chroma.from_documents(docs, embeddings)


def generate_content(query, call_transcript):
    # relevant_docs = db_chroma.similarity_search(query)
    system_prompt = f"""You are a professional writer of motivational letters.\
You will be given a content from a knowledge base below, delimited by triple \
backticks. Your job is to use knowledge from this data and write a \
motivational letter for graduate school application. Only write content \
using data from the knowledgebase, do not claim facts from outside of it. \
Make the letter very personal with regards to the knowledge base.

Knowledge Base: ```{call_transcript}```
"""
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=query)
    message = [system_message, human_message]
    response = llm(message)
    return response.content


system_session_prompt = """As a professional writer of motivational letters, \
your task is to write a sales proposal provided to you according to \
the required changes. You will make the recommended changes to the \
sales proposal and return the entire proposal with thse changes. \
Your job depends on the answers you provide so play close attention to \
the queries you recieve.
"""


def main():
    st.title("ChatGPT 🤖 Powered Chatbot")
    st.header("Sales Proposal Generator")

    uploaded_file = st.file_uploader("Upload a word file", type="docx")
    if "messages" not in st.session_state:
        st.session_state.messages = [AIMessage(content="How can I help you?")]
    if uploaded_file is not None:
        # extract text from word file
        call_transcript = docx2txt.process(uploaded_file)
        # load_into_chroma(call_transcript)

        with st.sidebar:
            user_input = st.text_area("Enter your query: ", key="user_input")
            st.session_state.messages.append(HumanMessage(content=user_input))

            if st.button("Generate content"):
                with st.spinner("GPT is thinking..."):
                    response = generate_content(user_input, call_transcript)
                    st.session_state.messages.append(AIMessage(content=response))

    # display message history
    messages = st.session_state.get('messages', [])
    for i in range(len(messages)):
        if i % 2 == 0:
            message(messages[i].content, is_user=False, key=str(i) + '_user')
        else:
            message(messages[i].content, is_user=True, key=str(i) + '_ai')


if __name__ == '__main__':
    main()
