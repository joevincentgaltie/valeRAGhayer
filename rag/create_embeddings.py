from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#import faiss

from langchain.vectorstores import FAISS


loader = CSVLoader('../data/VALERIE_HAYER.csv', metadata_columns=["source","source_date"],encoding='utf-8')
documents =loader.load()

#CHunking the documents with Recursive Character Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

db = FAISS.from_documents(chunked_docs, embedding_function)
db.save_local("../data/faiss_index")

