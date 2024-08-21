from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
import os

class Vectorstore:
    def __init__(self, embeddings, save=False):
        self.embeddings = embeddings
        self.save = save

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def generate_vectorstore(self, pdf_docs, store_name, chunk_size=2500, chunk_overlap=300):
        raw_text = self.get_pdf_text(pdf_docs=pdf_docs)
        text_chunks = self.get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if os.path.exists(f"./vectorstore/{store_name}.pkl") and self.save == True:
            with open(f"./vectorstore/{store_name}.pkl", "rb") as f: 
                vectorstore = pickle.load(f) 
        else: 
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
            if self.save:
                os.makedirs('./vectorstore/', exist_ok=True)
                with open(f"./vectorstore/{store_name}.pkl", "wb") as f:
                    pickle.dump(vectorstore, f) 
        return vectorstore
