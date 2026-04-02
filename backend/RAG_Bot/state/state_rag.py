import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from backend.legal_research import source_rank_bonus
from backend.paths import STATELAWS_DIR, VECTOR_DB_DIR

PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "state_db")
DATA_DIRECTORY = STATELAWS_DIR
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
logger = logging.getLogger(__name__)
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return _vectorstore


def has_state_corpus() -> bool:
    return DATA_DIRECTORY.exists() and any(DATA_DIRECTORY.rglob("*.txt"))


def state_rag_answer(query: str, state: str | None = None, matter_type: str | None = None) -> dict:
    if not state or not has_state_corpus():
        return {
            "answer": "",
            "confidence": 0.0,
            "chunks": [],
            "matches": [],
        }

    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(
            query,
            k=TOP_K,
            filter={"applicable_state": state},
        )
        reranked = []
        for doc, score in results:
            reranked.append(
                (doc, score + source_rank_bonus(doc.metadata or {}, state=state, matter_type=matter_type, query=query))
            )
        reranked.sort(key=lambda item: item[1])
        results = reranked[:TOP_K]
    except Exception as e:
        logger.error("State RAG retrieval error: %s", e)
        return {"answer": "", "confidence": 0.0, "chunks": [], "matches": []}

    if not results:
        logger.info("State RAG: no results for state=%r query=%r", state, query)
        return {"answer": "", "confidence": 0.0, "chunks": [], "matches": []}

    top_doc, score = results[0]
    confidence = 1 / (1 + score)
    chunks = [doc.page_content.strip() for doc, _ in results if doc.page_content.strip()]
    matches = [doc.metadata for doc, _ in results]
    context = "\n\n".join([f"Source {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    prompt = f"""You are a Legal AI Assistant specialized in Indian law.
Rules:
1. Answer only from the provided state-specific context.
2. Focus on material applicable to {state}.
3. Do not use outside knowledge.
4. If the context does not clearly answer the question, say exactly: I don't have enough information in the provided context.
5. Keep the answer short and factual.

User query: {query}

Context:
{context}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.2,
        do_sample=False,
        repetition_penalty=1.05,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "answer": answer,
        "confidence": float(confidence),
        "chunks": chunks,
        "matches": matches,
        "top_document_metadata": top_doc.metadata,
    }
