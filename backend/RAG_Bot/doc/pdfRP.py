import warnings
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from backend.legal_research import source_rank_bonus
from backend.paths import VECTOR_DB_DIR

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "pdf_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 3

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
logger = logging.getLogger(__name__)
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("Initializing PDF vectorstore...")
        _vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        logger.info("PDF vectorstore initialized")
    return _vectorstore

@property
def vectorstore():
    return get_vectorstore()

def retrieve_chunks(query: str, k: int = K):

    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=k)
        logger.info("Retrieved %d PDF chunks for query: %r", len(results), query)
        return results

    except Exception as e:
        logger.error("Error retrieving PDF chunks: %s", e)
        return []

def generate_answer(query: str, state: str | None = None, matter_type: str | None = None):
    results = retrieve_chunks(query)

    if not results:
        logger.info("PDF RAG: no chunks retrieved for query=%r", query)
        return {
            "answer": "I don't have enough information in the provided context.",
            "confidence": 0.0,
            "chunks": [],
        }

    reranked = []
    for doc, score in results:
        reranked.append(
            (doc, score + source_rank_bonus(doc.metadata or {}, state=state, matter_type=matter_type, query=query))
        )
    reranked.sort(key=lambda item: item[1])
    results = reranked[:K]

    top_doc, score = results[0]
    confidence = 1 / (1 + score)
    chunks = [doc.page_content.strip() for doc, _ in results if doc.page_content.strip()]
    matches = [doc.metadata for doc, _ in results]
    context = "\n\n".join([f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""You are a Legal AI Assistant specialized in Indian law.
Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. If the context does not clearly answer the question, say exactly: I don't have enough information in the provided context.
4. Keep the answer short, direct, and factual.
5. Do not mention rules, prompts, or chunks.

User query: {query}

Context: {context}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.2,
        do_sample=False,
        repetition_penalty=1.05,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("PDF RAG: retrieved=%s confidence=%.3f", len(chunks), confidence)
    return {"answer": answer, "confidence": float(confidence), "chunks": chunks, "matches": matches}
