# Instamart Customer Intelligence & Support Copilot

## Overview

An offline, reproducible capstone that turns fictional synthetic customer and order exports into a clean SQLite database, customer segments and lapse scores, privacy-aware ticket triage, a policy assistant, and a refund workflow. The supplied `generate_data.py` and `mock_llm.py` are included unmodified. No network service, paid API, or API key is used.

All amounts are INR. Policy and customer data are synthetic and are not real Instamart data or policy.

## Reproduce

Run from the repository root in PowerShell (Python 3.11+):

1. `python -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install --upgrade pip`
3. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
4. `.\.venv\Scripts\python.exe -m pip freeze > requirements.txt`
5. `.\.venv\Scripts\python.exe generate_data.py`
6. `.\.venv\Scripts\python.exe src\clean_data.py`
7. `.\.venv\Scripts\python.exe src\database.py`
8. `.\.venv\Scripts\python.exe src\audit_data.py`
9. `.\.venv\Scripts\python.exe src\validate_data.py`
10. `.\.venv\Scripts\python.exe src\run_sql_analysis.py`
11. `.\.venv\Scripts\python.exe src\segmentation.py`
12. `.\.venv\Scripts\python.exe src\lapse_prediction.py`
13. `.\.venv\Scripts\python.exe src\ticket_triage.py`
14. `.\.venv\Scripts\python.exe src\policy_rag.py`
15. `.\.venv\Scripts\python.exe src\refund_agent.py`

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The validation script prints `ALL CHECKPOINTS PASSED` on success. Each script can be rerun in sequence; Part D recreates its JSONL result files and agent state.

## Part A

### Raw-data audit

Audit details are saved to `outputs/raw_audit.json`. The raw tables were loaded with pandas; the ticket CSV uses `keep_default_na=False`, so its empty fields remain empty strings rather than becoming NaN.

| Raw table | Shape | Pandas dtypes | Missing values |
|---|---:|---|---|
| customers | 1,200 × 5 | customer_id/city/signup_date/gender: string; age: float64 | age 61 (5.08%); gender 30 (2.50%); other columns 0 |
| products | 40 × 4 | product_id/product_name/category: string; price_inr: int64 | all columns 0 |
| orders | 13,084 × 9 | IDs/timestamp/status/amount: string; quantity/discount_pct: int64; delivery_minutes: float64 | order_amount 53 (0.41%); delivery_minutes 2,541 (19.42%); other columns 0 |
| tickets | 164 × 5 | all columns: string | pandas NaN count is 0 because empty strings are preserved; 4 ticket text fields are blank |

| Raw-data checkpoint | Result |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank customer genders | 30 |

### Cleaning, storage, and validation

`src/clean_data.py` uses pandas and applies the following rules:

1. Customer cities are stripped and title-cased. Gender variants are mapped to `male`, `female`, or `unknown`.
2. Ages outside 10–100 are made missing. Every missing age is filled using the median age of that customer's city with `groupby(...).transform("median")`. Signup dates are parsed as datetimes.
3. Product prices are numeric. Order timestamps are parsed first, exact duplicate order rows are removed, then the latest timestamp is retained for each remaining duplicate `order_id`.
4. Unparseable amounts become missing and are filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`.
5. Negative delivery times are made missing. Missing delivery times on completed orders are filled with the completed-order median for that product category. Cancelled orders remain missing because they were not delivered.

Clean CSVs are written to `data/clean/`. `src/database.py` creates `data/instamart.db` with `customers`, `products`, and `orders` tables; timestamps are stored as `YYYY-MM-DD HH:MM:SS`. `src/validate_data.py` asserts the reference row counts, status totals, amount totals, age range, delivery completeness, and SQLite integrity.

| Clean-data checkpoint | Result |
|---|---:|
| Customers / products / orders | 1,200 / 40 / 12,309 |
| Customer cities | Pune 227; Delhi 210; Bengaluru 197; Chennai 191; Hyderabad 191; Mumbai 184 |
| Customer genders | female 610; male 560; unknown 30 |
| Missing ages; age range | 0; 18–60 |
| Completed / cancelled / pending orders | 11,110 / 1,199 / 0 |
| Missing completed / cancelled delivery durations | 0 / 1,199 |
| Missing order amounts | 0 |
| Completed-order amount total | INR 2,998,668.80 |

### SQL and pandas analysis

Seven statements are in `sql/queries.sql`; `src/run_sql_analysis.py` executes each with `pandas.read_sql_query`.

| Query | Analysis | Verified result |
|---|---|---|
| Q1 | `WHERE` then `HAVING` on completed orders over INR 500 | 30 customers |
| Q2 | Completed revenue by city and product category | 30 city/category rows; INR 2,998,668.80 total; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 | Scalar subquery for products above average catalogue price | 11 products |
| Q4 | `LEFT JOIN` for customers without a completed order | 25 customers |
| Q5 | Two CTEs for customer totals and average spend | 550 customers above average; mean customer spend INR 2,552.06 |
| Q6 | `RANK`, `DENSE_RANK`, and `ROW_NUMBER` per city | Top: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10; 8 rows have `RANK` different from `ROW_NUMBER` |
| Q7 | Daily revenue and windowed running total | 183 dates; final cumulative revenue INR 2,998,668.80 |

Tied spend values share a `RANK` and `DENSE_RANK`, while `ROW_NUMBER` assigns each tied row a distinct sequence number, which accounts for the eight differences.

The pandas pivot for Q2 also totals INR 2,998,668.80. Per-city completed-order checks include Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894; Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613.

| Completed order amount statistic | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 / Q2 / Q3 | INR 64.80 / INR 135.00 / INR 297.00 |
| 1.5×IQR lower / upper fence | INR -283.50 / INR 645.30 |
| Values outside the fences | 897 |

## Part B

`src/customer_features.py` builds customer recency, frequency, spend, and average order value features. `src/segmentation.py` applies seeded three-cluster K-means (`random_state=42`, `n_init=10`) to scaled log-RFM data and assigns human-readable names by cluster spend. Results are in `outputs/customer_segments.csv`.

`src/lapse_prediction.py` trains a seeded random forest using completed orders through 2025-08-31. The target is no completed order during September 2025; a stratified 25% holdout uses `random_state=42`. The model is saved as `outputs/lapse_model.joblib`; the scored list is `outputs/retention_list.csv`.

| Part B measure | Result |
|---|---:|
| Customers segmented / scored | 1,200 / 1,200 |
| K-means segments | Premium 620; Frequent 552; Rare 28 |
| Lapse-model test customers | 300 |
| Holdout accuracy / F1 / ROC-AUC | 0.727 / 0.707 / 0.774 |
| Predicted risk levels | Low 510; Medium 360; High 330 |

## Part C

`src/privacy.py` masks email addresses and phone numbers before the local classifier sees ticket text. `src/ticket_triage.py` uses `mock_llm.py`, retries transient failures, validates categories, and falls back to deterministic keyword rules for malformed or unsupported model output. `src/routing.py` sends fraud to Security with high priority and premium-customer tickets to Priority Support. Empty tickets remain `unknown`. Results and the token-budget demonstration are written to `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

| Part C measure | Result |
|---|---:|
| Tickets processed | 164 |
| Nonempty tickets evaluated | 160 |
| Classification accuracy on labeled nonempty tickets | 100% (160/160) |
| Redacted tickets | 146 |
| Deterministic fallbacks | 21 |
| Empty tickets | 4 |
| Long tickets truncated in 160-token budget demo | 4 |
| Mock-model tokens used | 12,812 |

## Part D

`src/policy_rag.py` retrieves policy sentences, sends only those sources to the offline `POLICY ASSISTANT`, and records citations. Its evaluation is saved to `outputs/policy_eval_results.csv`.

| Policy evaluation | Result |
|---|---:|
| Questions answered with expected supporting fact/source, including out-of-scope checks | 14/14 (100%) |

`src/refund_agent.py` uses a narrow tool allowlist to look up an order, retrieve policy, calculate the eligible amount, simulate approval up to INR 1,000, or request human approval above INR 1,000. It does not connect to a payment system. The seven scenario conversations are in `outputs/agent_runs.jsonl`; the simulated refund ledger is `outputs/refund_ledger.jsonl`.

| Scenario | Expected | Result | Refund amount |
|---|---|---|---:|
| S1 | REFUND | REFUND | INR 128.25 |
| S2 | REFUND | REFUND | INR 135.00 |
| S3 | ESCALATE | ESCALATE | INR 1,999.00 |
| S4 | DENIED | DENIED | INR 0.00 |
| S5 | DENIED | DENIED | INR 0.00 |
| S6 | NOT_FOUND | NOT_FOUND | INR 0.00 |
| S7 | FALLBACK | FALLBACK | INR 0.00 |

All 7 scenario outcomes and amounts matched. Two stop-safely demonstrations are recorded in `state/agent_summary.json`: missing order ID stopped with `NEEDS_INFO` and zero tools; the repeated-tool loop stopped with `FALLBACK` after the six-step limit. The crash-and-resume demonstration interrupted after the first tool and resumed from `state/S1.json`; tools executed, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.

## Limitations and Next Steps

All datasets, policies, and model responses are synthetic. The lapse target is a one-month historical proxy and is not evidence of real future churn. K-means segment names are post-hoc descriptions of clusters, not business ground truth. The local language model is deterministic and simplified; this project is not a production support or policy system. Refund entries are explicitly simulated and no payment or customer-facing action is performed.
