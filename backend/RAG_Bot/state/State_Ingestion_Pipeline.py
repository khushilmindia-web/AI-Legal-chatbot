from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.RAG_Bot.txt.Txt_Ingestion_Pipeline import clean_text, split_by_section, split_faq_qa, HEADER_RE
from backend.paths import STATELAWS_DIR, VECTOR_DB_DIR

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "state_db")
DOCS_PATH = STATELAWS_DIR

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def parse_state_file(file_path: Path) -> list:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw_text)
    if not text:
        return []

    documents = split_by_section(file_path, text) if HEADER_RE.search(text) else split_faq_qa(file_path, text)
    state_name = file_path.parent.name if file_path.parent != DOCS_PATH else ""

    for document in documents:
        document.metadata["jurisdiction_scope"] = "state"
        document.metadata["applicable_state"] = state_name
        document.metadata["applicable_states_text"] = state_name
    return documents


def load_documents() -> list:
    documents = []
    if not DOCS_PATH.exists():
        return documents

    for file_path in sorted(DOCS_PATH.rglob("*.txt")):
        documents.extend(parse_state_file(file_path))
    return documents


def main() -> None:
    documents = load_documents()
    if not documents:
        print("No state-law documents found. Skipping state-db ingestion.")
        return

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"State-law vector store created at {PERSIST_DIRECTORY}")
