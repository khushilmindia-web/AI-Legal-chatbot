import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.RAG_Bot.txt.Txt_Ingestion_Pipeline import clean_text
from backend.paths import CASELAW_DIR, VECTOR_DB_DIR

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "case_db")
DOCS_PATH = CASELAW_DIR

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
CASE_HEADER_RE = re.compile(r"^(.*?vs\.?.*?)$", re.IGNORECASE | re.MULTILINE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
STATE_NAMES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry",
]


def infer_state_from_court(court: str) -> str:
    lowered = court.lower()
    for state_name in STATE_NAMES:
        if state_name.lower() in lowered:
            return state_name
    if "bombay" in lowered:
        return "Maharashtra"
    if "madras" in lowered:
        return "Tamil Nadu"
    if "calcutta" in lowered:
        return "West Bengal"
    return ""


def parse_case_file(file_path: Path) -> list[Document]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw_text)
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else file_path.stem
    case_name = first_line if CASE_HEADER_RE.search(first_line) else file_path.stem
    year_match = YEAR_RE.search(text[:500])
    year = year_match.group(0) if year_match else ""
    court = file_path.parent.name if file_path.parent != DOCS_PATH else "Unknown Court"
    applicable_state = infer_state_from_court(court)

    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "doc_type": "case_law",
        "title": case_name,
        "case_name": case_name,
        "court": court,
        "year": year,
        "citation": file_path.stem,
        "act_name": "Case Law",
        "jurisdiction_scope": "case_law",
        "applicable_state": applicable_state,
        "applicable_states_text": applicable_state,
    }
    return [Document(page_content=text, metadata=metadata)]


def load_documents() -> list[Document]:
    documents: list[Document] = []
    if not DOCS_PATH.exists():
        return documents

    for file_path in sorted(DOCS_PATH.rglob("*.txt")):
        documents.extend(parse_case_file(file_path))
    return documents


def main() -> None:
    documents = load_documents()
    if not documents:
        print("No case-law documents found. Skipping case-db ingestion.")
        return

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Case-law vector store created at {PERSIST_DIRECTORY}")
