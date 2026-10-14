import re, sys
import anthropic

from finpilot.config import LLM_MODEL
from finpilot.retrieval.rerank import search, id2chunk

client = anthropic.Anthropic()

SYSTEM = """You are a financial news analyst. Answer the question using ONLY the provided news excerpts.
Cite every claim with the excerpt ID in square brackets, like [1847_0].
If the excerpts don't contain the answer, say "I don't have enough information to answer."
Do not use outside knowledge."""

def build_context(results):
    blocks = []
    for r in results:
        c = id2chunk[r[0]]
        blocks.append(f"[{c['chunk_id']}] ({c['date']}, {c['ticker']}) {c['title']}\n{c['text']}")
    return "\n\n".join(blocks)

def answer(question, k=10):
    results = search(question, k=k)
    context = build_context(results)
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=SYSTEM,
        extra_body={"temperature": 0},
        messages=[{"role": "user", "content": f"News excerpts:\n\n{context}\n\nQuestion: {question}"}],
    )
    text = msg.content[0].text
    cited = set(re.findall(r"\[(\d+_\d+)\]", text))
    retrieved = {r[0] for r in results}
    return {
        "answer": text,
        "citations": sorted(cited),
        "invalid_citations": sorted(cited - retrieved),
        "sources": [r[0] for r in results],
        "context": context,
    }

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How did Alcoa perform in the third quarter?"
    out = answer(q)
    print(out["answer"])
    print("\ncitations:", out["citations"])
    print("invalid:", out["invalid_citations"])