import os
import glob
import warnings
import logging
from pathlib import Path

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma

from dotenv import load_dotenv
from backend.paths import PDFS_DIR, VECTOR_DB_DIR
load_dotenv()

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(docs_path: str | Path = PDFS_DIR):
    print(f"Loading PDF documents from {docs_path}...")

    pattern = os.path.join(str(docs_path), "**", "*.pdf")
    all_files = glob.glob(pattern, recursive=True)

    if not all_files:
        raise FileNotFoundError(f"No PDF files found in {docs_path}. Please add .pdf files.")

    total_files = len(all_files)
    print(f"Found {total_files} PDF files to process.")

    documents = []
    success_count = 0
    error_files = []

    for idx, file_path in enumerate(tqdm(all_files, desc="Processing PDFs", unit="file"), start=1):

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            success_count += 1

            if idx % 10 == 0:
                print(f"\n Loaded {success_count} files so far...")

        except Exception as e:
            error_files.append((file_path, str(e)))

    print(f"\nFinished loading PDF files:")
    print(f"Total PDF files processed: {total_files}")

    print(f"Successfully loaded: {success_count}")
    print(f"Failed: {len(error_files)}")

    if error_files:
        print("\nErrors encountered:")

        for f, err in error_files[:5]: 
            print(f" {os.path.basename(f)}: {err}")

        if len(error_files) > 5:
            print(f" ... and {len(error_files)-5} more errors")

    print(f"Total documents (pages) loaded: {len(documents)}")

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents could be loaded from {docs_path}. Check your PDF files and permissions.")
    return documents

def split_documents(documents, chunk_size=1500, chunk_overlap=150):
    print("\n Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)    
    print(f" Created {len(chunks)} chunks")

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1} ")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content[:150]}...")

        if len(chunks) > 5:
            print(f"\n...and {len(chunks) - 5} more chunks")
    return chunks 

def create_vector_store(chunks, persist_directory: str | Path = VECTOR_DB_DIR / "pdf_db"):
    print("\n Creating embeddings and storing in ChromaDB...")
    
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"})

    batch_size = 400
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f" Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print("\n Finished creating vector store")
    print(f" Vector store saved at: {persist_directory}")
    return vectorstore

def main():
    print("Main")
    documents = load_documents(docs_path=PDFS_DIR)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
