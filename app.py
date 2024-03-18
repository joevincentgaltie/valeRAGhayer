import streamlit as st
from rag.prompt_generation import * 
from langchain.embeddings import HuggingFaceEmbeddings

st.title('ValeRAG Hayer')

db = FAISS.load_local("data/faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), allow_dangerous_deserialization=True)
retriever = db.as_retriever()



prompt = st.chat_input("Pose moi une question sur les activités de Valérie Hayer au Parlement")
if prompt:
    st.write(f"{prompt}")
    st.write("")
    st.write(f"{question_answering(prompt, retriever)}")
