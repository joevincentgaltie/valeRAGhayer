from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
#import faiss

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("API_MISTRAL")
print(api_key)

loader = CSVLoader('../data/all_french_explanations.csv',metadata_columns=["party", "number","name", "source", "source_date","orientation"], encoding="utf-8")
documents =loader.load()
print(len(documents))



#CHunking the documents with Recursive Character Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(documents)

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
db = Chroma.from_documents(chunked_docs, embeddings, persist_directory="../chroma_db")


