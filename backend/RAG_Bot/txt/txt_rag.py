import logging
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from backend.legal_research import source_rank_bonus
from backend.paths import VECTOR_DB_DIR

persistent_directory = str(VECTOR_DB_DIR / "doc_db")
TOP_K = 5

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
logger = logging.getLogger(__name__)

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("Initializing TXT vectorstore...")
        _vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        logger.info("TXT vectorstore initialized")
    return _vectorstore

@property
def vectorstore():
    return get_vectorstore()

SECTION_QUERY_RE = re.compile(r"\bsection\s+([0-9]+[A-Z]?)\b", re.IGNORECASE)
ARTICLE_QUERY_RE = re.compile(r"\barticle\s+([0-9]+[A-Z]?)\b", re.IGNORECASE)
RULE_QUERY_RE = re.compile(r"\brule\s+([0-9]+[A-Z]?)\b", re.IGNORECASE)

def extract_section_number(query: str) -> str | None:
    match = SECTION_QUERY_RE.search(query)
    if not match:
        return None
    return match.group(1).upper()

def extract_article_number(query: str) -> str | None:
    match = ARTICLE_QUERY_RE.search(query)
    if not match:
        return None
    return match.group(1).upper()

def extract_rule_number(query: str) -> str | None:
    match = RULE_QUERY_RE.search(query)
    if not match:
        return None
    return match.group(1).upper()

def retrieve_documents(query: str, k: int = TOP_K):
    try:
        vs = get_vectorstore()
        section_number = extract_section_number(query)
        article_number = extract_article_number(query)
        rule_number = extract_rule_number(query)
        if section_number:
            filtered_results = vs.similarity_search_with_score(
                query,
                k=k,
                filter={"section_number": section_number},
            )
            if filtered_results:
                logger.info("Retrieved %d filtered documents for section %s", len(filtered_results), section_number)
                return filtered_results, f"section:{section_number}"

        if article_number:
            filtered_results = vs.similarity_search_with_score(
                query,
                k=k,
                filter={"article_number": article_number},
            )
            if filtered_results:
                logger.info("Retrieved %d filtered documents for article %s", len(filtered_results), article_number)
                return filtered_results, f"article:{article_number}"

        if rule_number:
            filtered_results = vs.similarity_search_with_score(
                query,
                k=k,
                filter={"rule_number": rule_number},
            )
            if filtered_results:
                logger.info("Retrieved %d filtered documents for rule %s", len(filtered_results), rule_number)
                return filtered_results, f"rule:{rule_number}"

        results = vs.similarity_search_with_score(query, k=k)
        logger.info("Retrieved %d documents for query: %r (section=%s article=%s rule=%s)", len(results), query, section_number, article_number, rule_number)
        return results, None

    except Exception as e:
        logger.error("Error retrieving documents: %s", e)
        return [], None

def retrieve_documents_for_state(query: str, state: str | None, k: int = TOP_K):
    results, specific_ref = retrieve_documents(query, k=k)
    if not state or not results:
        reranked = rerank_results(results, query=query, state=state)
        return reranked[:k], specific_ref

    prioritized = rerank_results(results, query=query, state=state)
    logger.info("Applied state-aware prioritization for state=%r", state)
    return prioritized[:k], specific_ref

def rerank_results(results, *, query: str, state: str | None = None, matter_type: str | None = None):
    reranked = []
    for doc, score in results:
        metadata = doc.metadata or {}
        score_adjustment = source_rank_bonus(
            metadata,
            state=state,
            matter_type=matter_type,
            query=query,
        )
        reranked.append((doc, score + score_adjustment))
    reranked.sort(key=lambda item: item[1])
    return reranked

def build_context(results) -> tuple[list[str], str]:
    chunks: list[str] = []
    context_blocks: list[str] = []

    for index, (doc, _) in enumerate(results, start=1):
        content = doc.page_content.strip()
        if not content:
            continue

        chunks.append(content)
        label = doc.metadata.get("title") or doc.metadata.get("file_name") or f"Result {index}"
        if doc.metadata.get("article_number"):
            label = f"Article {doc.metadata['article_number']} - {label}"
        elif doc.metadata.get("section_number"):
            label = f"Section {doc.metadata['section_number']} - {label}"
        elif doc.metadata.get("rule_number"):
            label = f"Rule {doc.metadata['rule_number']} - {label}"
        elif doc.metadata.get("chapter_number"):
            label = f"Chapter {doc.metadata['chapter_number']} - {label}"

        citation_hint = doc.metadata.get("citation_hint")
        if citation_hint:
            label = f"{label} [{citation_hint}]"

        context_blocks.append(f"Source {index}: {label}\n{content}")
    return chunks, "\n\n".join(context_blocks)

def txt_rag_answer(query: str, state: str | None = None, matter_type: str | None = None) -> dict:
    results, section_number = retrieve_documents(query, k=TOP_K)
    results = rerank_results(results, query=query, state=state, matter_type=matter_type)[:TOP_K]
    if not results:
        logger.info("TXT RAG: no blocks retrieved for query=%r", query)
        return {
            "answer": "I don't have enough information in the provided context.",
            "confidence": 0.0,
            "chunks": [],
            "matches": [],
        }

    top_doc, score = results[0]
    confidence = 1 / (1 + score)
    chunks, context = build_context(results)
    matches = [doc.metadata for doc, _ in results]

    if section_number:
        if section_number.startswith("section:"):
            section_filter_instruction = f"The user explicitly asked about Section {section_number.split(':',1)[1]}. Prefer that section."
        elif section_number.startswith("article:"):
            section_filter_instruction = f"The user explicitly asked about Article {section_number.split(':',1)[1]}. Prefer that article."
        elif section_number.startswith("rule:"):
            section_filter_instruction = f"The user explicitly asked about Rule {section_number.split(':',1)[1]}. Prefer that rule."
        else:
            section_filter_instruction = "Use the most relevant retrieved blocks."
    else:
        section_filter_instruction = "Use the most relevant retrieved blocks."

    prompt = f"""You are a Legal AI Assistant specialized in Indian law.
Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. If the context does not clearly answer the question, say exactly: I don't have enough information in the provided context.
4. Keep the answer short, direct, and factual.
5. Do not mention prompts, retrieval, metadata, or source numbers.
6. If the query names an article, section, or rule, prioritize that reference and do not answer from unrelated provisions.
7. Where the context contains a citation hint, prefer the cited provision directly.

User query:{query}

Instruction: {section_filter_instruction}

Context: {context}

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
    logger.info(
        "TXT RAG: retrieved=%s confidence=%.3f section_filter=%s",
        len(chunks),
        confidence,
        section_number or "none",
    )
    return {
        "answer": answer,
        "confidence": float(confidence),
        "chunks": chunks,
        "matches": matches,
        "top_document_metadata": top_doc.metadata,
    }
