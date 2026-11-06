import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from database import DATABASE, ROOT
from policy_rag import answer_question


sys.path.insert(0, str(ROOT))
import mock_llm


OUTPUTS = ROOT / "outputs"
STATE = ROOT / "state"
MAX_STEPS = 6
AGENT_SYSTEM = (
    "AGENT. Check a refund request using get_order, search_policy, calculate_refund, "
    "issue_refund, and escalate_to_human tools. Never invent order data."
)
TOOL_SCHEMAS = [
    {"name": name, "description": name.replace("_", " ")}
    for name in (
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
        "escalate_to_human",
    )
]


class SimulatedCrash(Exception):
    pass


def tool_get_order(arguments, customer_id):
    order_id = arguments.get("order_id", "")
    with sqlite3.connect(DATABASE) as connection:
        row = connection.execute(
            """
            SELECT order_id, customer_id, product_id, status, order_amount, delivery_minutes
            FROM orders WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order_not_found", "order_id": order_id}
    order = dict(
        zip(
            ("order_id", "customer_id", "product_id", "status", "order_amount", "delivery_minutes"),
            row,
        )
    )
    if customer_id and order["customer_id"] != customer_id:
        return {"error": "order_not_found", "order_id": order_id}
    return order


def tool_search_policy(arguments):
    result = answer_question(arguments.get("query", "refund policy"))
    if not result["sources"]:
        return {"error": "no_relevant_policy"}
    return {"sources": result["sources"], "policy_excerpt": result["answer"]}


def tool_calculate_refund(arguments):
    try:
        amount = float(arguments["order_amount"])
        delivery = int(arguments["delivery_minutes"])
    except (KeyError, TypeError, ValueError):
        return {"error": "missing_order_measurement", "refund_inr": 0.0}
    refund = round(amount, 2) if delivery > 45 else 0.0
    return {"refund_inr": refund}


def write_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_dispatch(name, arguments, scenario_id, ledger, customer_id):
    if name == "get_order":
        return tool_get_order(arguments, customer_id)
    if name == "search_policy":
        return tool_search_policy(arguments)
    if name == "calculate_refund":
        return tool_calculate_refund(arguments)
    if name == "issue_refund":
        refund = round(float(arguments["refund_inr"]), 2)
        write_jsonl(
            ledger,
            {
                "scenario_id": scenario_id,
                "order_id": arguments["order_id"],
                "refund_inr": refund,
                "status": "SIMULATED_APPROVAL",
                "side_effect": "no payment was issued",
            },
        )
        return {"refund_inr": refund, "status": "simulated_approval"}
    if name == "escalate_to_human":
        return {
            "order_id": arguments["order_id"],
            "refund_inr": round(float(arguments["refund_inr"]), 2),
            "status": "human_approval_required",
        }
    return {"error": "tool_not_allowed", "tool": name}


def checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(path)


def load_scenario_state(path, user_message, scenario_id):
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["scenario_id"] == scenario_id and state["user_message"] == user_message:
            return state
    return {
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "tools_executed": [],
        "steps": 0,
    }


def run_agent(
    scenario_id,
    user_message,
    ledger_path,
    state_path,
    customer_id="",
    crash_after_tools=None,
):
    state = load_scenario_state(state_path, user_message, scenario_id)
    state["customer_id"] = customer_id
    tools_executed = state["tools_executed"]
    while state["steps"] < MAX_STEPS:
        response = mock_llm.chat(state["messages"], tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        if response.choices[0].finish_reason != "tool_calls" or not message.tool_calls:
            state["final"] = message.content or ""
            break
        call = message.tool_calls[0]
        name = call.name
        arguments = call.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result = tool_dispatch(
            name, arguments, scenario_id, ledger_path, state["customer_id"]
        )
        state["messages"].append(
            {"role": "tool", "name": name, "content": json.dumps(result)}
        )
        tools_executed.append(name)
        state["steps"] += 1
        checkpoint(state_path, state)
        if crash_after_tools is not None and len(tools_executed) >= crash_after_tools:
            raise SimulatedCrash(f"simulated interruption after {name}")
    else:
        state["final"] = "FALLBACK: safe handoff after reaching the tool-step limit."

    if state["steps"] >= MAX_STEPS and not state.get("final"):
        state["final"] = "FALLBACK: safe handoff after reaching the tool-step limit."
    checkpoint(state_path, state)
    return state["final"], list(tools_executed), state["steps"]


def outcome_from_response(response):
    if response.startswith("REFUND:"):
        return "REFUND"
    if response.startswith("ESCALATE:"):
        return "ESCALATE"
    if response.startswith("DENIED:"):
        return "DENIED"
    if response.startswith("NOT_FOUND:"):
        return "NOT_FOUND"
    if response.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"
    if response.startswith("FALLBACK:"):
        return "FALLBACK"
    return "ERROR"


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    ledger = OUTPUTS / "refund_ledger.jsonl"
    runs_path = OUTPUTS / "agent_runs.jsonl"
    for path in (ledger, runs_path):
        path.write_text("", encoding="utf-8")
    scenarios = pd.read_csv(ROOT / "data" / "raw" / "agent_scenarios.csv")
    results = []
    crash_resume_tools = []
    for index, row in enumerate(scenarios.to_dict(orient="records")):
        scenario_id = row["scenario_id"]
        state_path = STATE / f"{scenario_id}.json"
        mock_llm.reset()
        crash_after = 1 if index == 0 else None
        try:
            final, tool_names, steps = run_agent(
                scenario_id,
                row["message"],
                ledger,
                state_path,
                customer_id=row["customer_id"],
                crash_after_tools=crash_after,
            )
        except SimulatedCrash:
            final, tool_names, steps = run_agent(
                scenario_id,
                row["message"],
                ledger,
                state_path,
                customer_id=row["customer_id"],
            )
        if index == 0:
            crash_resume_tools = tool_names

        outcome = outcome_from_response(final)
        amount = 0.0
        if outcome == "REFUND":
            try:
                amount = float(final.split("INR", 1)[1].split()[0])
            except (IndexError, ValueError):
                amount = 0.0
        elif outcome == "ESCALATE":
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            escalation = next(
                (
                    json.loads(message["content"])
                    for message in reversed(saved_state["messages"])
                    if message.get("role") == "tool"
                    and message.get("name") == "escalate_to_human"
                ),
                {},
            )
            amount = float(escalation.get("refund_inr", 0.0))
        record = {
            "scenario_id": scenario_id,
            "customer_id": row["customer_id"],
            "expected_outcome": row["expected_outcome"],
            "actual_outcome": outcome,
            "expected_refund_inr": float(row["expected_refund_inr"]),
            "actual_refund_inr": amount,
            "response": final,
            "tools_executed": tool_names,
            "tool_steps": steps,
            "matches_expected": outcome == row["expected_outcome"]
            and round(amount, 2) == round(float(row["expected_refund_inr"]), 2),
        }
        results.append(record)
        write_jsonl(runs_path, record)

    safe_stops = []
    for demo_id, message in (
        ("stop_missing_order", "Can I get a refund?"),
        ("stop_loop", "[LOOP] Refund order ORD-100001 please."),
    ):
        mock_llm.reset()
        response, tool_names, steps = run_agent(
            demo_id,
            message,
            ledger,
            STATE / f"{demo_id}.json",
        )
        safe_stops.append(
            {
                "demo": demo_id,
                "outcome": outcome_from_response(response),
                "steps": steps,
                "tools_executed": tool_names,
                "response": response,
            }
        )
    summary = {
        "scenarios": len(results),
        "passed": sum(result["matches_expected"] for result in results),
        "crash_resume_tools": crash_resume_tools,
        "safe_stops": safe_stops,
    }
    (STATE / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
