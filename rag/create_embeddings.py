from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
#import faiss

from dotenv import load_dotenv
import os 
load_dotenv()
API_MISTRAL = os.getenv("API_MISTRAL")


loader = CSVLoader('../data/all_french_explanations.csv',metadata_columns=["party", "number","name", "source", "source_date","orientation"], encoding="utf-8")
documents =loader.load()


documents = [doc for doc in documents if doc.metadata["party"] == "Groupe Renew Europe"]

#CHunking the documents with Recursive Character Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(documents)

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_MISTRAL)
db = Chroma.from_documents(documents, embeddings, persist_directory="../chroma_db", collection_name="Explanations")


