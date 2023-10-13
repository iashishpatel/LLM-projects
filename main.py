import os
import time

import streamlit as st
from langchain.llms import OpenAI
import pickle
from env import OPENAI_API_KEY
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import SeleniumURLLoader


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_click = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"
main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
if process_url_click:
    ### Load Data
    loader = SeleniumURLLoader(urls=urls)
    main_placefolder.text("Data Loading.....Started..🔧🔧🔧")
    data = loader.load()
    # st.write(data)
    ### Split Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitting.....Started..🔧🔧🔧")
    docs = text_splitter.split_documents(data)
    #creat embedding and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Building....🔧🔧🔧")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= vectorstore.as_retriever())
            result = chain({"question": query},return_only_outputs=True)

            st.header("Answer")
            st.subheader(result["answer"])

        #Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)