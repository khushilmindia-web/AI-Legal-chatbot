import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.paths import DOCS_DIR, VECTOR_DB_DIR

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "doc_db")
DOCS_PATH = str(DOCS_DIR)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

HEADER_RE = re.compile(
    r"^(Section|Article|Rule|Chapter)\s+([0-9]+[A-Z]?)\s*[-:\.]?\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FAQ_QUESTION_RE = re.compile(
    r"^(?:q[\.\):]?\s*)?.+?\?\s*$",
    re.IGNORECASE,
)
INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

DOMAIN_KEYWORDS = {
    "constitutional": ["constitution", "article", "fundamental rights", "writ"],
    "criminal": ["ipc", "criminal", "fir", "bail", "offence", "police"],
    "civil": ["cpc", "civil procedure", "injunction", "suit"],
    "family": ["family", "marriage", "divorce", "maintenance", "custody"],
    "property": ["property", "land", "partition", "rent", "tenancy"],
    "labour": ["labour", "industrial", "wages", "employment", "gratuity"],
    "consumer": ["consumer", "deficiency", "refund"],
    "contract": ["contract", "agreement", "consideration", "specific performance"],
    "tort": ["tort", "negligence", "defamation"],
    "evidence": ["evidence", "proof", "witness"],
    "human_rights": ["human rights", "liberty", "dignity"],
    "banking": ["banking", "insurance"],
    "environment": ["environment", "pollution", "forest"],
    "cyber": ["cyber", "computer", "online", "digital"],
}

def load_txt_files(docs_path: str = DOCS_PATH) -> list[Path]:
    base_path = Path(docs_path)
    txt_files = sorted(base_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")
    return txt_files

def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" -:\t")

def normalize_act_name(file_path: Path) -> str:
    name = file_path.stem.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name

def infer_source_type(file_path: Path, text: str) -> str:
    haystack = f"{file_path.stem} {text[:2500]}".lower()
    if "constitution" in haystack or "article " in haystack:
        return "constitution"
    if "rules" in file_path.stem.lower() or "rule " in haystack:
        return "state_rule" if detect_states(haystack) else "rule"
    if "notification" in haystack or "order" in haystack or "circular" in haystack:
        return "government_order"
    return "statute"

def build_citation_hint(kind: str | None, number: str | None, act_name: str, title: str | None = None) -> str:
    if kind and number:
        return f"{act_name} - {kind.title()} {number}" + (f" ({title})" if title else "")
    return act_name

def detect_domain(file_path: Path, text: str) -> str:
    haystack = f"{file_path.stem} {text[:2000]}".lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return domain
    return "general"

def detect_states(text: str) -> list[str]:
    lowered = text.lower()
    matches = [state for state in INDIAN_STATES_AND_UTS if state.lower() in lowered]
    return matches[:5]

def detect_jurisdiction_scope(file_path: Path, text: str) -> tuple[str, list[str]]:
    state_matches = detect_states(f"{file_path.stem}\n{text[:4000]}")
    if state_matches:
        return "state", state_matches
    return "central", []

def build_base_metadata(file_path: Path, text: str) -> dict:
    jurisdiction_scope, applicable_states = detect_jurisdiction_scope(file_path, text)
    act_name = normalize_act_name(file_path)
    source_type = infer_source_type(file_path, text)
    return {
        "source": str(file_path),
        "file_name": file_path.name,
        "act_name": act_name,
        "legal_domain": detect_domain(file_path, text),
        "jurisdiction_scope": jurisdiction_scope,
        "applicable_state": applicable_states[0] if applicable_states else "",
        "applicable_states_text": ", ".join(applicable_states),
        "source_type": source_type,
        "authority_level": "primary" if source_type in {"constitution", "statute", "state_rule", "rule"} else "secondary",
        "citation_hint": build_citation_hint(None, None, act_name),
    }

def make_section_document(file_path: Path, kind: str, number: str, title: str, content: str, base_metadata: dict) -> Document:
    section_id = number.upper()
    clean_content = re.sub(r"\s+", " ", content).strip()
    block_text = f"{kind.title()} {section_id} - {title}\n{clean_content}".strip()

    metadata = {
        **base_metadata,
        "doc_type": kind.lower(),
        "title": title,
        "reference_kind": kind.lower(),
        "citation_hint": build_citation_hint(kind, section_id, base_metadata["act_name"], title),
    }
    if kind.lower() == "section":
        metadata["section_number"] = section_id
    elif kind.lower() == "article":
        metadata["article_number"] = section_id
    elif kind.lower() == "rule":
        metadata["rule_number"] = section_id
    elif kind.lower() == "chapter":
        metadata["chapter_number"] = section_id
    return Document(page_content=block_text, metadata=metadata)

def split_by_section(file_path: Path, text: str) -> list[Document]:
    documents: list[Document] = []
    base_metadata = build_base_metadata(file_path, text)
    current_kind = None
    current_number = None
    current_title = None
    current_lines: list[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        match = HEADER_RE.match(line)
        if match:
            if current_kind and current_number and current_title:
                documents.append(
                    make_section_document(
                        file_path=file_path,
                        kind=current_kind,
                        number=current_number,
                        title=current_title,
                        content=" ".join(current_lines),
                        base_metadata=base_metadata,
                    )
                )
            current_kind = match.group(1)
            current_number = match.group(2)
            current_title = normalize_title(match.group(3))
            current_lines = []
            continue

        if current_kind:
            current_lines.append(line)

    if current_kind and current_number and current_title:
        documents.append(
            make_section_document(
                file_path=file_path,
                kind=current_kind,
                number=current_number,
                title=current_title,
                content=" ".join(current_lines),
                base_metadata=base_metadata,
            )
        )

    return documents

def make_faq_document(file_path: Path, question: str, answer: str, index: int, base_metadata: dict) -> Document:
    question = re.sub(r"\s+", " ", question).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    block_text = f"{question}\n{answer}".strip()
    return Document(

        page_content=block_text,
        metadata={
            **base_metadata,
            "doc_type": "faq",
            "title": question,
            "faq_index": index,
            "source_type": "faq",
            "authority_level": "secondary",
            "citation_hint": build_citation_hint(None, None, base_metadata["act_name"], question),
        },
    )

def split_faq_qa(file_path: Path, text: str) -> list[Document]:
    documents: list[Document] = []
    base_metadata = build_base_metadata(file_path, text)
    current_question = None
    current_answer_lines: list[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if FAQ_QUESTION_RE.match(line):
            if current_question and current_answer_lines:
                documents.append(

                    make_faq_document(
                        file_path=file_path,
                        question=current_question,
                        answer=" ".join(current_answer_lines),
                        index=len(documents) + 1,
                        base_metadata=base_metadata,
                    )
                )
            current_question = line
            current_answer_lines = []
            continue

        if current_question:
            current_answer_lines.append(line)

    if current_question and current_answer_lines:
        documents.append(

            make_faq_document(
                file_path=file_path,
                question=current_question,
                answer=" ".join(current_answer_lines),
                index=len(documents) + 1,
                base_metadata=base_metadata,
            )
        )

    if documents:
        return documents

    fallback_text = re.sub(r"\s+", " ", text).strip()
    if not fallback_text:
        return []

    return [
        Document(
            page_content=fallback_text,

            metadata={
                **base_metadata,
                "doc_type": "plain_text",
                "title": file_path.stem,
                "source_type": "plain_text",
                "authority_level": "secondary",
            },
        )
    ]


def parse_txt_file(file_path: Path) -> list[Document]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw_text)
    if not text:
        return []

    if HEADER_RE.search(text):
        return split_by_section(file_path, text)
    return split_faq_qa(file_path, text)

def load_documents(docs_path: str = DOCS_PATH) -> list[Document]:
    print(f"Loading documents from {docs_path}...")
    all_documents: list[Document] = []

    for file_path in load_txt_files(docs_path):
        parsed_documents = parse_txt_file(file_path)
        all_documents.extend(parsed_documents)
        print(f"{file_path.name}: loaded {len(parsed_documents)} semantic blocks")

    if not all_documents:
        raise ValueError("No semantic blocks were created from the .txt files.")

    for i, doc in enumerate(all_documents[:5], start=1):
        print(f"\n--- block {i} ---")
        print(f"source: {doc.metadata.get('source')}")
        print(f"type: {doc.metadata.get('doc_type')}")
        print(f"title: {doc.metadata.get('title')}")
        safe_content = doc.page_content[:400].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(safe_content)

    return all_documents

def create_vector_store(documents: list[Document], persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main() -> None:
    documents = load_documents(docs_path=DOCS_PATH)
    create_vector_store(documents)

if __name__ == "__main__":
    main()
