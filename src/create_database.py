import os
import shutil
import warnings
from typing import List, Optional

# Suppress PyPDF warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
warnings.filterwarnings("ignore", message=".*CryptographyDeprecationWarning.*")
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentManager:
    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.CHROMA_PATH = "./chroma"
        self.DATA_PATH = "./data"

    def load_docs(self) -> List[Document]: 
        loader = PyPDFDirectoryLoader(
            path = self.DATA_PATH,
            glob = "*.pdf"
        )
        docs = loader.load()
        return docs

    def split_text(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size = 512,
            chunk_overlap = 50,
            add_start_index = True
        )
        chunks = splitter.split_documents(docs)
        
        print(f"split {len(docs)} documents into {len(chunks)} chunks")
        return chunks

    def embed_and_store_docs(self,chunks: List[Document]):
        # Clear out the database first.
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)
            
        print("Starting embedding process...")
        Chroma.from_documents(
            documents = chunks,
            collection_name = "the_documents",
            persist_directory = self.CHROMA_PATH,
            embedding = self.embedding        
            )
        print(f"Saved {len(chunks)} chunks to {self.CHROMA_PATH}.")

    def load_vector_store(self):
        try:
            db = Chroma(
                collection_name = "the_documents",
                persist_directory = self.CHROMA_PATH,
                embedding_function = self.embedding   
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
        count = db._collection.count()
        if count == 0:
            print("Collection is empty")
        return db

if __name__ == "__main__":
    document_manager = DocumentManager()
    docs = document_manager.load_docs()
    chunks = document_manager.split_text(docs)
    document_manager.embed_and_store_docs(chunks)
    db = document_manager.load_vector_store()
    print(db._collection.count())

