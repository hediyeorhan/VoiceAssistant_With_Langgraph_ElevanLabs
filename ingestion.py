from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langgraph.checkpoint.memory import MemorySaver

import glob

import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

file_path = os.getenv("PDF_PATH")

# .pdf uzantılı tüm dosyaları yükle
loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=int(os.getenv("CHUNK_SIZE")), chunk_overlap=int(os.getenv("CHUNK_OVERLAP"))
)
doc_splits = text_splitter.split_documents(documents)
print(doc_splits)
#doc_splits = format_docs(doc_splits)

memory = MemorySaver()

vectorstore = Chroma.from_documents(
     documents=doc_splits,
     collection_name="rag-chroma",
     embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
     persist_directory="./.chroma_vector_db",
 )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma_vector_db",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
).as_retriever()
