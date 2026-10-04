import json
from rag import answer, client, MODEL

JUDGE = """You are checking a RAG answer for faithfulness.

Excerpts:
{context}

Answer:
{answer}

Is every factual claim in the answer directly supported by the excerpts?
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""

REFUSAL = "enough information"

rows = [json.loads(l) for l in open("eval/rag_questions.jsonl")]
s = {"ans": 0, "unans": 0, "invalid": 0, "cited": 0, "gold": 0,
     "false_refusal": 0, "correct_refusal": 0, "faithful": 0}

for r in rows:
    out = answer(r["question"])
    refused = REFUSAL in out["answer"].lower()
    if out["invalid_citations"]:
        s["invalid"] += 1
        print("[INVALID CITE]", r["question"])
    if not r["answerable"]:
        s["unans"] += 1
        s["correct_refusal"] += refused
        if not refused:
            print("[SHOULD REFUSE]", r["question"])
        continue
    s["ans"] += 1
    if refused:
        s["false_refusal"] += 1
        print("[FALSE REFUSAL]", r["question"])
        print("   gold retrieved:", r["gold"] in out["sources"])
        continue
    s["cited"] += bool(out["citations"])
    s["gold"] += r["gold"] in out["citations"]
    verdict = client.messages.create(
        model=MODEL,
        max_tokens=5,
        extra_body={"temperature": 0},
        messages=[{"role": "user", "content": JUDGE.format(context=out["context"], answer=out["answer"])}],
    ).content[0].text
    if verdict.strip().upper().startswith("SUPPORTED"):
        s["faithful"] += 1
    else:
        print("[UNFAITHFUL]", r["question"])

answered = s["ans"] - s["false_refusal"]
print(f"\nInvalid citation rate: {s['invalid'] / len(rows):.0%}")
print(f"Correct refusal rate:  {s['correct_refusal'] / s['unans']:.0%}")
print(f"False refusal rate:    {s['false_refusal'] / s['ans']:.0%}")
print(f"Citation coverage:     {s['cited'] / answered:.0%}")
print(f"Gold cited rate:       {s['gold'] / s['ans']:.0%}")
print(f"Faithfulness:          {s['faithful'] / answered:.0%}")