import json, random
import anthropic
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL = "claude-haiku-4-5-20251001"
client = anthropic.Anthropic()
random.seed(7)

chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
sample = random.sample(chunks, 20)

PROMPT = """Write ONE specific factual question that can be answered using only the news excerpt below.
Include the company name and the time period (quarter, year, or date) so the question is clear without seeing the excerpt.
Output only the question.

Date: {date}
Title: {title}
Excerpt: {text}"""

UNANSWERABLE = [
    "What was Tesla's total revenue in 2024?",
    "What is Alcoa's earnings guidance for fiscal year 2026?",
    "How much did Ford's EV unit lose in 2024?",
    "What was Bitcoin's price on January 1, 2025?",
    "Who became CEO of Starbucks in 2024?",
]

with open("eval/rag_questions.jsonl", "w") as f:
    for c in sample:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": PROMPT.format(date=c["date"], title=c["title"], text=c["text"])}],
        )
        q = msg.content[0].text.strip()
        f.write(json.dumps({"question": q, "gold": c["chunk_id"], "answerable": True}) + "\n")
    for q in UNANSWERABLE:
        f.write(json.dumps({"question": q, "gold": None, "answerable": False}) + "\n")

print("wrote", len(sample) + len(UNANSWERABLE), "questions")