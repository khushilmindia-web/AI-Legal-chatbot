import json
from app import generate_llm_answer, invalidate_llm_cache

def main() -> None:
    query = "What should I do after receiving a legal notice for a consumer dispute in India?"
    context = "RAG confidence was below threshold. Provide a concise general answer with practical next steps."

    invalidate_llm_cache(query, context)

    first = generate_llm_answer(query, context)
    second = generate_llm_answer(query, context)

    print("First call:")
    print(json.dumps(first, indent=2))
    print("\nSecond call:")
    print(json.dumps(second, indent=2))

    if not first.get("answer"):
        raise SystemExit("LLM fallback test failed: empty answer")
    if not second.get("cached"):
        raise SystemExit("LLM fallback cache test failed: second call was not cached")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
