import os
import shutil
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.paths import DOCS_DIR, FRONTEND_DIR, VECTOR_DB_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOC_DB_PATH = VECTOR_DB_DIR / "doc_db"
PDF_DB_PATH = VECTOR_DB_DIR / "pdf_db"
STATE_DB_PATH = VECTOR_DB_DIR / "state_db"
CASE_DB_PATH = VECTOR_DB_DIR / "case_db"

def cleanup_vector_stores():
    logger.info("Cleaning up old vector stores...")
    
    try:
        import gc
        gc.collect()
    except:
        pass
    
    for db_path in [DOC_DB_PATH, PDF_DB_PATH, STATE_DB_PATH, CASE_DB_PATH]:
        if os.path.exists(db_path):
            logger.info(f"Removing {db_path}")
            
            try:
                for attempt in range(3):
            
                    try:
                        import stat
                        def handle_remove_error(func, path, exc):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)                        
                        shutil.rmtree(db_path, onerror=handle_remove_error)
                        logger.info(f" Removed {db_path}")
                        break

                    except PermissionError:
                        if attempt < 2:
                            logger.info(f"Retry {attempt + 1}...")
                            import time
                            time.sleep(0.5)
            
                        else:
                            logger.warning(f"Could not remove {db_path} (in use). Skipping...")
                            break
            
            except Exception as e:
                logger.warning(f" Could not remove {db_path}: {e}")
        else:
            logger.info(f"  {db_path} does not exist (OK)")

def verify_documents_exist():
    logger.info(" Checking for documents...")
    
    if not os.path.exists(DOCS_DIR):
        logger.error(f"ERROR: {DOCS_DIR} not found!")
        return False
    
    txt_files = list(Path(DOCS_DIR).glob("*.txt"))
    if not txt_files:
        logger.error(f"ERROR: No .txt files found in {DOCS_DIR}")
        return False
    
    logger.info(f"  Found {len(txt_files)} documents")
    for i, file in enumerate(txt_files[:5], 1):
        logger.info(f"    {i}. {file.name}")
    
    if len(txt_files) > 5:
        logger.info(f" ... and {len(txt_files) - 5} more")
    
    return True

def ingest_documents():
    logger.info(" Ingesting documents...")
    
    try:
        from backend.RAG_Bot.txt.Txt_Ingestion_Pipeline import main as ingest_txt
        from backend.RAG_Bot.state.State_Ingestion_Pipeline import main as ingest_state
        from backend.RAG_Bot.case.Case_Ingestion_Pipeline import main as ingest_case
        logger.info("  Starting TXT document ingestion...")
        ingest_txt()
        logger.info("  Starting state-law ingestion...")
        ingest_state()
        logger.info("  Starting case-law ingestion...")
        ingest_case()
        logger.info(" TXT ingestion completed")
        return True
    
    except Exception as e:
        logger.error(f" Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_vector_store():
    logger.info("Verifying vector store...")
    
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=str(DOC_DB_PATH), embedding_function=embeddings)
        
        count = vectorstore._collection.count()
        logger.info(f"  Vector store contains {count} documents")
        
        if count == 0:
            logger.warning(" Vector store is EMPTY!")
            return False
        
        logger.info(f"Vector store verified with {count} documents")
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main():
    logger.info("=" * 80)
    logger.info("LEGAL AI CHATBOT INITIALIZATION")
    logger.info("=" * 80)
    
    if not verify_documents_exist():
        logger.error("Initialization failed: No documents found")
        return False
    
    cleanup_vector_stores()
    
    if not ingest_documents():
        logger.error("Initialization failed: Document ingestion failed")
        return False
    
    if not verify_vector_store():
        logger.error("Initialization failed: Vector store verification failed")
        return False
    
    logger.info("INITIALIZATION SUCCESSFUL")
    logger.info("\nYour chatbot is ready! Documents have been indexed.")

    logger.info("You can now:")
    logger.info("  - Run: python App.py")
    logger.info("  - Access: http://127.0.0.1:8000 (API)")
    logger.info("  - Frontend directory: %s", FRONTEND_DIR)

    logger.info("\nDebug endpoints:")
    logger.info("  - GET http://127.0.0.1:8000/debug/status (Check vector stores)")
    logger.info("  - GET http://127.0.0.1:8000/debug/ingest (Manual re-ingestion)")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
