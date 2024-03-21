from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#import faiss

from langchain.vectorstores import FAISS


loader = CSVLoader('../data/all_french_explanations.csv',metadata_columns=["party", "number","name", "source", "source_date","orientation"], encoding="utf-8")
documents =loader.load()


documents = [doc for doc in documents if doc.metadata["party"] == "Groupe Renew Europe"]

#CHunking the documents with Recursive Character Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

db = FAISS.from_documents(chunked_docs, embedding_function)
db.save_local("../data/faiss_index_groupe_renew_europe")


