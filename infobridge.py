import os
from dotenv import load_dotenv

load_dotenv()



import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


# using Ollma
from langchain_ollama import OllamaLLM
Model1=OllamaLLM(model="llama3.2")

# get the embeddings
from langchain_ollama import OllamaEmbeddings
embed=OllamaEmbeddings(model="llama3.2")


st.title("InfoBridge-The Summary Tool")
st.sidebar.title("News Artical urls")


urls=[]

url=st.sidebar.text_input(f"Enter the url")
urls.append(url)

URl_processing_done=st.sidebar.button("Process URLs")

main_placeholder=st.empty()
if URl_processing_done:
    main_placeholder.text("Data loading started....................")
    loader=UnstructuredURLLoader(urls=urls)
    
    data=loader.load()
    main_placeholder.text("Data loading done ✅....................")

    #splitting the data
    main_placeholder.text("Text splitting started ....................")
    text_splitter=RecursiveCharacterTextSplitter(separators=["\n\n","\n","."],chunk_size=1000,chunk_overlap=200)
    
    docs=text_splitter.split_documents(data)
    main_placeholder.text("Text splitting done ✅ ....................")
    # storing them vector store
    main_placeholder.text("Vectore embedding started ....................")
    vector_store=FAISS.from_documents(embedding=embed,documents=docs)
    main_placeholder.text("Vectore embedding done ✅....................")
    # saving them to local file
    vector_store.save_local("vector_index")
    main_placeholder.text("Save to local file ✅")

query=main_placeholder.text_input("Enter your query")
if query:
    load_vector_store=FAISS.load_local(folder_path="vector_index",embeddings=embed,allow_dangerous_deserialization=True)
    chain=RetrievalQA.from_llm(llm=Model1,retriever=load_vector_store.as_retriever())
    result = chain({"query": query}, return_only_outputs=True)
    st.header("Here is my answer")
    st.subheader(result["result"])



    