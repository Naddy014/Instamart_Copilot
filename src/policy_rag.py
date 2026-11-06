import re
import sys
from pathlib import Path

import pandas as pd

from database import ROOT


sys.path.insert(0, str(ROOT))
import mock_llm


POLICIES = ROOT / "data" / "policies"
OUTPUTS = ROOT / "outputs"
POLICY_SYSTEM = (
    "POLICY ASSISTANT. Answer only from the provided policy CONTEXT. "
    "If the answer is not supported, say that the policy documents do not contain it."
)
WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "the", "a", "an", "is", "of", "to", "for", "in", "on", "how", "what",
    "when", "who", "are", "do", "does", "i", "my", "be", "can", "after",
    "from", "it", "or", "and", "at", "by", "which", "why", "if", "than", "that",
}


def terms(text):
    return {
        word[:5].lower()
        for word in WORD_PATTERN.findall(text)
        if word.lower() not in STOPWORDS
    }


def load_sentences():
    rows = []
    for path in sorted(POLICIES.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        for sentence in re.split(r"(?<=[.!?])\s+", content.strip()):
            if sentence:
                rows.append((path.name, sentence.strip()))
    if not rows:
        raise FileNotFoundError(f"No policy documents found in {POLICIES}")
    return rows


def retrieve(question, sentences, limit=8):
    query = terms(question)
    if not query:
        return []
    frequencies = {
        term: sum(term in terms(sentence) for _, sentence in sentences)
        for term in query
    }
    scored = []
    for filename, sentence in sentences:
        overlap = query & terms(sentence)
        score = sum(1 / frequencies[term] for term in overlap)
        if score:
            scored.append((score, filename, sentence))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(filename, sentence) for _, filename, sentence in scored[:limit]]


def answer_question(question, sentences=None):
    policy_sentences = sentences or load_sentences()
    retrieved = retrieve(question, policy_sentences)
    if not retrieved:
        return {
            "answer": "I could not find that information in the policy documents.",
            "sources": [],
            "tokens": 0,
            "retrieved": [],
        }
    context = "\n".join(f"[{filename}] {sentence}" for filename, sentence in retrieved)
    messages = [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION: {question}"},
    ]
    response = mock_llm.chat(messages)
    answer = response.choices[0].message.content or ""
    sources = sorted(
        {
            filename
            for filename, sentence in retrieved
            if sentence in answer or answer in sentence
        }
    )
    return {
        "answer": answer,
        "sources": sources,
        "tokens": response.usage.total_tokens,
        "retrieved": [filename for filename, _ in retrieved],
    }


def evaluate():
    evaluation = pd.read_csv(ROOT / "data" / "raw" / "policy_eval.csv")
    sentences = load_sentences()
    rows = []
    for row in evaluation.to_dict(orient="records"):
        result = answer_question(row["question"], sentences)
        answer = result["answer"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]
        found = (
            expected_doc == "NONE"
            and not result["sources"]
            or expected_doc in result["sources"]
            and expected_fact.casefold() in answer.casefold()
        )
        rows.append(
            {
                "question": row["question"],
                "expected_doc": expected_doc,
                "expected_fact": expected_fact,
                "answer": answer,
                "sources": ";".join(result["sources"]),
                "correct": found,
                "tokens": result["tokens"],
            }
        )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUTS / "policy_eval_results.csv", index=False)
    accuracy = float(frame["correct"].mean())
    print(f"policy evaluation: {int(frame['correct'].sum())}/{len(frame)} ({accuracy:.1%})")
    return frame


if __name__ == "__main__":
    evaluate()
