import json
import re
import sys
import time

import pandas as pd

from database import ROOT
from privacy import redact_pii
from routing import route_ticket


sys.path.insert(0, str(ROOT))
import mock_llm


OUTPUTS = ROOT / "outputs"
ALLOWED = {"billing", "delivery", "technical", "fraud"}
SYSTEM = (
    "CLASSIFY each support ticket as one JSON object with category and reason. "
    "Allowed categories: billing, delivery, technical, fraud. "
    "EXAMPLES: unauthorized transaction=fraud; app crash=technical; invoice=billing; "
    "rider late=delivery."
)


def fallback_category(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    rules = [
        ("fraud", {"unauthorized", "fraud", "hacked", "stolen"}),
        ("technical", {"app", "crash", "login", "otp", "error", "freezes"}),
        ("billing", {"charged", "invoice", "bill", "refund", "deducted", "payment"}),
        ("delivery", {"arrived", "rider", "delivery", "delivered", "late"}),
    ]
    return next((category for category, keywords in rules if words & keywords), "technical")


def classify(text):
    if not text.strip():
        return "unknown", "empty ticket", 0, "empty"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    response_tokens = 0
    for attempt in range(3):
        try:
            response = mock_llm.chat(messages)
            response_tokens += response.usage.total_tokens
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            category = parsed.get("category")
            if category in ALLOWED:
                return category, str(parsed.get("reason", "")), response_tokens, "model"
            break
        except mock_llm.TransientLLMError:
            if attempt == 2:
                break
            time.sleep(0)
        except (json.JSONDecodeError, AttributeError, TypeError):
            break
    category = fallback_category(text)
    return category, "deterministic fallback after invalid model response", response_tokens, "fallback"


def run():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(ROOT / "data" / "raw" / "tickets.csv", keep_default_na=False)
    segments_path = OUTPUTS / "customer_segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError("Run src/segmentation.py before ticket triage.")
    segments = pd.read_csv(segments_path)[["customer_id", "segment"]]
    tickets = tickets.merge(segments, on="customer_id", how="left")
    mock_llm.reset()

    results = []
    budget_rows = []
    for row in tickets.to_dict(orient="records"):
        safe_text, redactions = redact_pii(row["text"])
        category, reason, tokens, source = classify(safe_text)
        team, priority = route_ticket(category, row["segment"])
        valid_text = bool(row["text"].strip())
        results.append(
            {
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "category": category,
                "reason": reason,
                "priority": priority if valid_text else "Low",
                "route": team,
                "customer_segment": row["segment"],
                "pii_redacted": bool(redactions["emails"] or redactions["phones"]),
                "email_redactions": redactions["emails"],
                "phone_redactions": redactions["phones"],
                "classification_source": source,
                "token_usage": tokens,
                "true_category": row["true_category"],
            }
        )
        original_tokens = mock_llm.est_tokens(row["text"])
        capped = " ".join(safe_text.split()[:120])
        budget_rows.append(
            {
                "ticket_id": row["ticket_id"],
                "original_tokens": original_tokens,
                "budget_tokens": 160,
                "submitted_tokens": mock_llm.est_tokens(capped),
                "truncated": len(capped) < len(safe_text),
                "empty_ticket": not valid_text,
            }
        )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(OUTPUTS / "triage_results.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
    evaluated = result_frame.loc[result_frame["true_category"].ne("none")]
    accuracy = float(evaluated["category"].eq(evaluated["true_category"]).mean())
    counts = result_frame["category"].value_counts().to_dict()
    summary = {
        "tickets": len(result_frame),
        "evaluated_nonempty_tickets": len(evaluated),
        "accuracy": accuracy,
        "category_counts": counts,
        "redacted_tickets": int(result_frame["pii_redacted"].sum()),
        "fallbacks": int(result_frame["classification_source"].eq("fallback").sum()),
        "empty_tickets": int(result_frame["category"].eq("unknown").sum()),
        "budget_truncated_tickets": int(pd.DataFrame(budget_rows)["truncated"].sum()),
        "total_tokens": int(result_frame["token_usage"].sum()),
    }
    with (OUTPUTS / "triage_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
