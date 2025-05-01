import camelot 
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import numpy as np
import regex as re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
# from langchain_chroma import Chroma
from langchain.storage import InMemoryStore, InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
# import faiss
import uuid

ACCESS_TOKEN = "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"

def image_analysis(pdf_dir, page):
    """
        This function allows to debug a page with the help of plumber loader 
        Parameters : 1. page number of the document you want to see
                     2. pdf document directory
        returns the visualization of tables pdfplumber identified from the selected image
    """
    plumber_pages = pdfplumber.open(pdf_dir)
    return plumber_pages.pages[page - 1].to_image().debug_tablefinder()


def load_the_pages_in_plumber(pdf_dir):
    """
        Loads the pages in pdfplumber's loader
        Parameters : 1. pdf_dir : the path that leads to the pdf document
        returns pdfplumber loader object that can be used to get text, tables using pdfplumber
    """
    return pdfplumber.open(pdf_dir)

def get_slice_indices(page_str):
    """
        Function to convert string to a slicer, eg '1-20' converts to :20 to use in a list/array
        Parameter : page_str : A string with page numbers, has to have a hyphen in between
        returns a slice object
    """
    start, end = map(int, page_str.split('-'))
    return slice(start-1, end)

def get_all_the_tables(pdf_dir, pages='all'):
    """ 
        Function to get all the tables from selected number of pages, (or default all)
        parameters : pdf dir, pages : string object, either all, or can be 1-20 or 1,2,3,4,5
        note : takes a while
        returns : a tablelist object of n=number of tables
    """
    camelot_tables = camelot.read_pdf(pdf_dir, pages=pages)
    return camelot_tables

def get_text_from_pdf(pdf_dir, pages, document_class_flag=True):
    """ 
        Returns the list of document objects of raw text from selected pages from a pdf document,
        Parameters : pdf_dir, pages, document_class_flag : to determine whether you want a list of Document objects/ or just a list of strings.
    """
    pdf_docs = []
    plumb_loader = load_the_pages_in_plumber(pdf_dir)
    if pages == 'all':
        iterate_pages = plumb_loader.pages
    else:
        page_slicer = get_slice_indices(pages)
        iterate_pages = plumb_loader.pages[page_slicer]
    
    for page_num, page in enumerate(iterate_pages):
        text = page.extract_text()
        clean_text = cleanup_text(text)
        if document_class_flag:
            document = Document(page_content=clean_text,
                        metadata={"source": pdf_dir,
                                  "file_path" : pdf_dir,
                                  "page" : page_num + 1,  
                                  "total_pages" : len(plumb_loader.pages)
                                 }
                               )
            pdf_docs.append(document)
        else:
            pdf_docs.append(clean_text)
    
    return pdf_docs

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

def cleanup_text(text):
    """
        Cleans up the text, very basic to jsut remove the dots from the table of contents for now
        Parameters : ANY text in str format
        Returns a text without the combination of dots 
    """
    cleaned_text = text.replace("...", "")
    # clean_text = cleaned_text.replace("\n", " [NEWLINE] ")
    return cleaned_text

def cleanup_tables(df):
    df.columns = df.iloc[0]
    df.columns = [col.replace("\n", " ") for col in df.columns]
    df = df.iloc[1:]
    return df

def get_your_embedding(embedding_name="medembed"):
    """
    Function to get the type of embedding you want
    Parameters: a name for the particular embedding you want
    Returns the embedding function 
    """
    if embedding_name == 'medembed':
        class MedEmbedEmbeddings(Embeddings):
            def __init__(self, model_name="abhinand/MedEmbed-base-v0.1"):
                self.model = SentenceTransformer(model_name)
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True)
            def embed_query(self, text):
                return self.model.encode([text], convert_to_numpy=True)[0]
        medembed_embeddings = MedEmbedEmbeddings()
        return medembed_embeddings
    elif embedding_name == 'openai':
        embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-large")
        return embeddings_openai
    elif embedding_name == 'bge':
        embeddings_bge = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs = {"normalize_embeddings": True})
        return embeddings_bge
            
def get_retriever(id_key):
    medembed_embeddings = get_your_embedding()
    indexflatl2=faiss.IndexFlatL2(len(medembed_embeddings.embed_query("hello world")))
    vectorstore_faiss = FAISS(
        embedding_function=medembed_embeddings,
        index=indexflatl2,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    retriever = MultiVectorRetriever(
        vectorstore = vectorstore_faiss,
        docstore = InMemoryStore(),
        id_key = id_key
    )
    return retriever
    

def add_another_doc(pdf_dir, retriever, summary_chain, id_key):
    pdf_text = get_text_from_pdf(pdf_dir, 'all', True)
    pdf_tables = get_all_the_tables(pdf_dir)
    pdf_text_chunks= split_into_chunks(pdf_text)
    pdf_tables = [cleanup_tables(table.df) for table in pdf_tables]
    pdf_mmd_tables = [df.to_markdown(index=False) for df in pdf_tables]
    pdf_mmd_tables_summaries, pdf_text_chunks_summaries = summary_chain.batch(pdf_tables, {"max_concurrency" : 5}), summary_chain.batch(pdf_text_chunks, {"max_concurrency" : 5})
    # id_key = f"{pdf_dir.split('/')[-1]}"
    doc_ids = [str(uuid.uuid4()) for _ in pdf_text_chunks]
    table_ids = [str(uuid.uuid4()) for _ in pdf_mmd_tables]
    pdf_text_doc_summaries = [
        Document(page_content = s, metadata={id_key : doc_ids[i],
                                             "source_file" : pdf_dir,
                                             "page_num" : i + 1
                                            })
        for i, s in enumerate(pdf_text_chunks_summaries)
    ]
    pdf_table_doc_summaries = [
        Document(page_content = s, metadata={id_key : table_ids[i],
                                             "source_file" : pdf_dir,
                                             "table_num" : i + 1
                                            })
        for i, s in enumerate(pdf_mmd_tables_summaries)
    ]
    retriever.vectorstore.add_documents(pdf_text_doc_summaries)
    retriever.docstore.mset(list(zip(doc_ids, pdf_text_chunks)))
    retriever.vectorstore.add_documents(pdf_table_doc_summaries)
    retriever.docstore.mset(list(zip(table_ids, pdf_mmd_tables)))
    return 
                            
    
    
    
    

            
        
        

