from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader, DataFrameLoader, GitLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
import pandas as pd
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain_core.documents import Document
import os
import xml.etree.ElementTree as ET


class MedEmbedEmbeddings(Embeddings):
    def __init__(self, model_name="abhinand/MedEmbed-base-v0.1"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]




def the_docs_in_db(index_store, embed_fn):
    vector_db = FAISS.load_local(index_store, embed_fn, allow_dangerous_deserialization=True)
    seen = set()
    for doc in vector_db.docstore._dict.values():
        if doc.metadata['source_file'] in seen:
            continue
        else:
            seen.add(doc.metadata['source_file'])
    return seen

def split_into_chunks(pdf_docs):
    """
        Function that splits text into chunks so that LLM can eat easily
        Parameters: the docs in document objects
        Returns chunks of the text, x3 mul
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(pdf_docs)
    return text_chunks

def get_pdf_splits(pdf_file):

    loader = pdfplumber.open(pdf_file)
    docs = [Document(page_content = s.extract_text(), metadata={"page_num" : i + 1, "source_file" : pdf_file}) for i, s in enumerate(loader.pages)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

def get_excel_splits(file_path):
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unexpected file type")
    
    markdown_table = df.to_markdown()
    docs_mmd = Document(page_content=markdown_table, metadata={"source_file" : file_path})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    split_docs = text_splitter.split_documents([docs_mmd])
    return split_docs

def get_xml_splits(xml_file):
    loader = UnstructuredXMLLoader(xml_file)
    docs = loader.load_and_split()
    return docs

def embed_index(doc_list, embed_fn, index_store):
    try:
        temp_db = FAISS.from_documents(doc_list, embed_fn)
    except Exception as e:
        temp_db = FAISS.from_texts(doc_list, embed_fn)

    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn, allow_dangerous_deserialization=True)
        local_db.merge_from(temp_db)
        local_db.save_local(index_store)

    else:
        temp_db.save_local(folder_path=index_store)
        print(f"New store created called {index_store}")

