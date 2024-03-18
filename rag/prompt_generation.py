
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

import json
import pandas as pd

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
import requests



API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def extract_context(question, retriever):
    """
    Extract the context from the top 3 relevant documents
    """

    context = ''
    for i in range(3) : 
        context +=retriever.get_relevant_documents(question, k=3, source=True)[i].page_content
    return context

def templated_prompt(question, context):
    """
    Create a prompt for the chatbot
    """
    
    return f"Answer the following question in french based only on the provided context: {context} : {question}"

def question_answering(question, retriever) : 
    """
    Answer the question based on the context
    """
    context = extract_context(question, retriever)
    prompt = templated_prompt(question, context)
    len_prompt = len(prompt)


    return query({"inputs": prompt})[0]["generated_text"][len_prompt:]