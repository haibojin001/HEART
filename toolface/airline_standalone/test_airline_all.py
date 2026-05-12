"""
test_airline_all.py — TAU2 Airline 全工具 Benchmark 测试脚本

测试 14 个 airline 工具的 schema 是否能被 LLM 正确识别和调用。
所有测试数据均来自真实的 db.json。

用法：
  export ANTHROPIC_API_KEY=sk-ant-...
  python test_airline_all.py                        # 测全部 14 个工具
  python test_airline_all.py --model openai         # 使用 OpenAI
  python test_airline_all.py --tool search_direct_flight  # 只测一个
  python test_airline_all.py --domain read          # 只测 READ 类工具
  python test_airline_all.py --domain write         # 只测 WRITE 类工具
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 本地模块 ─────────────────────────────────────────────────────
from data_model import FlightDB
from tools import AirlineTools

DB_PATH = Path(__file__).parent / "db.json"
POLICY_PATH = Path(__file__).parent / "policy.md"


def load_tools() -> AirlineTools:
    db = FlightDB.load(str(DB_PATH))
    return AirlineTools(db)


def get_policy() -> str:
    if POLICY_PATH.exists():
        return POLICY_PATH.read_text()
    return "You are an airline customer service agent."


# ══════════════════════════════════════════════════════════════════
# 14 个工具的测试用例
# 所有 user_id / reservation_id / flight_number 均来自真实 db.json
# ══════════════════════════════════════════════════════════════════

TEST_CASES = {

    # ── READ tools ────────────────────────────────────────────────

    "get_user_details": {
        "type": "read",
        "query": "I am raj_sanchez_7340. Can you look up my account details?",
        "expected_tool": "get_user_details",
        "expected_args": {"user_id": "raj_sanchez_7340"},
        "note": "Basic user lookup",
    },

    "get_reservation_details": {
        "type": "read",
        "query": "I am raj_sanchez_7340. What are the details of my reservation Q69X3R?",
        "expected_tool": "get_reservation_details",
        "expected_args": {"reservation_id": "Q69X3R"},
        "note": "Economy round-trip, 3 flights",
    },

    "list_all_airports": {
        "type": "read",
        "query": "What airports are available in your system?",
        "expected_tool": "list_all_airports",
        "expected_args": {},
        "note": "No parameters needed",
    },

    "search_direct_flight": {
        "type": "read",
        "query": "Search for direct flights from PHL to LGA on 2024-05-16.",
        "expected_tool": "search_direct_flight",
        "expected_args": {"origin": "PHL", "destination": "LGA", "date": "2024-05-16"},
        "note": "HAT001 available on that date",
    },

    "search_onestop_flight": {
        "type": "read",
        "query": "Search for one-stop connecting flights from PHL to PHX on 2024-05-16.",
        "expected_tool": "search_onestop_flight",
        "expected_args": {"origin": "PHL", "destination": "PHX", "date": "2024-05-16"},
        "note": "PHL->LGA->PHX connection available",
    },

    "get_flight_status": {
        "type": "read",
        "query": "What is the status of flight HAT243 on 2024-05-20?",
        "expected_tool": "get_flight_status",
        "expected_args": {"flight_number": "HAT243", "date": "2024-05-20"},
        "note": "Flight in reservation Q69X3R",
    },

    # ── WRITE tools ───────────────────────────────────────────────

    "calculate": {
        "type": "generic",
        "query": "Calculate (450 + 162 + 166) * 1.1 for me.",
        "expected_tool": "calculate",
        "expected_args": {"expression": "(450 + 162 + 166) * 1.1"},
        "note": "Math expression evaluation",
    },

    "book_reservation": {
        "type": "write",
        # mia_li_3668: gold member, has credit_card_4421486, saved passenger Amelia Ahmed
        # HAT001 PHL->LGA on 2024-05-17, economy $189, 1 passenger = $189
        "query": (
            "I am mia_li_3668. Book a one-way economy flight from PHL to LGA on 2024-05-17, "
            "flight HAT001. One passenger: Amelia Ahmed, DOB 1957-03-21. "
            "Pay $189 with credit_card_4421486. No extra bags, no insurance."
        ),
        "expected_tool": "book_reservation",
        "expected_args": {
            "user_id": "mia_li_3668",
            "origin": "PHL",
            "destination": "LGA",
            "flight_type": "one_way",
            "cabin": "economy",
        },
        "note": "Gold member, credit card payment, 1 passenger",
    },

    "cancel_reservation": {
        "type": "write",
        # MZDDS4: raj_sanchez_7340, business cabin (can cancel without insurance)
        # created 2024-05-14, business = eligible for cancellation
        "query": (
            "I am raj_sanchez_7340. I want to cancel reservation MZDDS4. "
            "My plans have changed. I confirm I want to proceed."
        ),
        "expected_tool": "cancel_reservation",
        "expected_args": {"reservation_id": "MZDDS4"},
        "note": "Business cabin reservation, eligible for cancellation",
    },

    "update_reservation_baggages": {
        "type": "write",
        # 4WQ150: chen_jackson_3290, business, nonfree_baggages=0, gift_card_3576581 has $245
        # Adding 1 nonfree bag = $50
        "query": (
            "I am chen_jackson_3290. Add 1 non-free checked bag to reservation 4WQ150. "
            "Total bags should be 6 (5 existing + 1 new). Charge to gift_card_3576581. "
            "I confirm."
        ),
        "expected_tool": "update_reservation_baggages",
        "expected_args": {
            "reservation_id": "4WQ150",
            "total_baggages": 6,
            "nonfree_baggages": 1,
            "payment_id": "gift_card_3576581",
        },
        "note": "Business reservation, gift card payment, add 1 bag ($50)",
    },

    "update_reservation_flights": {
        "type": "write",
        # Q69X3R: raj_sanchez_7340, economy (not basic), future flights, can modify
        # Change to a different flight on same route
        # HAT001 PHL->LGA available on 2024-05-17
        "query": (
            "I am raj_sanchez_7340. I want to change the outbound flights on reservation Q69X3R "
            "to flight HAT001 on 2024-05-17 (PHL to LGA). Keep the return flight HAT206 on 2024-05-23. "
            "Keep economy cabin. Charge any difference to gift_card_4964153. I confirm."
        ),
        "expected_tool": "update_reservation_flights",
        "expected_args": {
            "reservation_id": "Q69X3R",
            "cabin": "economy",
            "payment_id": "gift_card_4964153",
        },
        "note": "Economy reservation, change outbound flight, keep return",
    },

    "update_reservation_passengers": {
        "type": "write",
        # 4WQ150: chen_jackson_3290, 3 passengers - update passenger info
        "query": (
            "I am chen_jackson_3290. Update the passengers on reservation 4WQ150. "
            "Keep the same 3 passengers but correct the DOB for Raj Smith to 1967-04-02. "
            "Passengers: Chen Jackson (1956-07-07), Raj Smith (1967-04-02), Fatima Martin (1970-01-20). "
            "I confirm."
        ),
        "expected_tool": "update_reservation_passengers",
        "expected_args": {
            "reservation_id": "4WQ150",
        },
        "note": "Update passenger DOB, must keep same count (3 passengers)",
    },

    "send_certificate": {
        "type": "write",
        # Compensation scenario: cancelled flight
        "query": (
            "I am raj_sanchez_7340. My flight was cancelled by the airline. "
            "Please send me a $100 travel certificate as compensation. "
            "I confirm."
        ),
        "expected_tool": "send_certificate",
        "expected_args": {
            "user_id": "raj_sanchez_7340",
            "amount": 100,
        },
        "note": "Compensation certificate, $100 for 1 passenger",
    },

    "transfer_to_human_agents": {
        "type": "generic",
        "query": (
            "I am raj_sanchez_7340. I need to speak with a human agent immediately. "
            "I have a complex issue with my reservation that cannot be resolved automatically."
        ),
        "expected_tool": "transfer_to_human_agents",
        "expected_args": {},  # summary arg - LLM generates this
        "note": "Explicit request for human agent",
    },

}


# ══════════════════════════════════════════════════════════════════
# Schema generation
# ══════════════════════════════════════════════════════════════════

def get_schemas_anthropic(tools: AirlineTools) -> list[dict]:
    schemas = []
    for name, tool in tools.get_tools().items():
        fn = tool.openai_schema["function"]
        schemas.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return schemas


def get_schemas_openai(tools: AirlineTools) -> list[dict]:
    return [tool.openai_schema for tool in tools.get_tools().values()]


# ══════════════════════════════════════════════════════════════════
# Tool execution
# ══════════════════════════════════════════════════════════════════

def execute_tool(tools: AirlineTools, tool_name: str, tool_input: dict) -> str:
    from datetime import date, datetime
    from pydantic import BaseModel

    def serialize(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, list):
            return [serialize(i) for i in obj]
        elif isinstance(obj, tuple):
            return [serialize(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    try:
        result = tools.use_tool(tool_name, **tool_input)
        return json.dumps(serialize(result), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# Single-turn tool call (no multi-turn conversation)
# We only need the FIRST tool call the LLM makes
# ══════════════════════════════════════════════════════════════════

def run_single_anthropic(query: str, tools: AirlineTools) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=get_policy(),
        tools=get_schemas_anthropic(tools),
        messages=[{"role": "user", "content": query}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return {
                "called": True,
                "name": block.name,
                "input": block.input,
                "id": block.id,
            }
    return {"called": False, "stop_reason": response.stop_reason}


def run_single_openai(query: str, tools: AirlineTools) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        tools=get_schemas_openai(tools),
        tool_choice="auto",
        messages=[
            {"role": "system", "content": get_policy()},
            {"role": "user", "content": query},
        ],
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {
            "called": True,
            "name": tc.function.name,
            "input": json.loads(tc.function.arguments),
            "id": tc.id,
        }
    return {"called": False, "finish_reason": response.choices[0].finish_reason}


# ══════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════

def verify_result(result: dict, expected_tool: str, expected_args: dict) -> tuple[bool, list[str]]:
    issues = []

    if not result.get("called"):
        return False, [f"No tool called (stop: {result.get('stop_reason') or result.get('finish_reason', '?')})"]

    # Check tool name
    if result["name"] != expected_tool:
        issues.append(f"Wrong tool: got '{result['name']}', expected '{expected_tool}'")
        return False, issues

    # Check expected args (subset match — LLM may add extra correct args)
    actual_input = result.get("input", {})
    for key, expected_val in expected_args.items():
        if key not in actual_input:
            issues.append(f"Missing arg '{key}'")
        elif expected_val is not None and actual_input[key] != expected_val:
            issues.append(f"Arg '{key}': got {actual_input[key]!r}, expected {expected_val!r}")

    return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════

def run_all(model: str = "anthropic",
            tool_filter: str = None,
            domain_filter: str = None):

    tools = load_tools()
    runner = run_single_anthropic if model == "anthropic" else run_single_openai

    # Filter test cases
    cases = {
        name: case for name, case in TEST_CASES.items()
        if (tool_filter is None or name == tool_filter)
        and (domain_filter is None or case["type"] == domain_filter)
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"airline_results_{model}_{timestamp}.json"

    print(f"\n{'='*65}")
    print(f"Airline Tool Benchmark — {model.upper()}  ({len(cases)} tools)")
    if tool_filter:   print(f"Tool filter:   {tool_filter}")
    if domain_filter: print(f"Domain filter: {domain_filter}")
    print(f"{'='*65}\n")

    results = []
    total = passed = failed = errored = 0

    for i, (tool_name, case) in enumerate(cases.items(), 1):
        total += 1
        print(f"[{i:02d}/{len(cases)}] {case['type'].upper():7} | {tool_name}")
        print(f"  query: {case['query'][:90]}{'...' if len(case['query'])>90 else ''}")
        print(f"  note:  {case['note']}")

        try:
            result = runner(case["query"], tools)
            ok, issues = verify_result(result, case["expected_tool"], case["expected_args"])

            if ok:
                passed += 1
                print(f"  ✅ Called: {result['name']}")
                preview = {k: v for i, (k, v) in enumerate(result["input"].items()) if i < 3}
                more = "..." if len(result["input"]) > 3 else ""
                print(f"  input: {json.dumps(preview, ensure_ascii=False)}{more}")
            else:
                failed += 1
                print(f"  ❌ FAIL")
                for issue in issues:
                    print(f"     → {issue}")
                if result.get("called"):
                    print(f"     actual input: {json.dumps(result.get('input',{}), ensure_ascii=False)[:150]}")

            results.append({
                "tool": tool_name,
                "type": case["type"],
                "ok": ok,
                "issues": issues,
                "result": result,
                "query": case["query"],
            })

        except Exception as e:
            errored += 1
            print(f"  ⚠️  Error: {e}")
            results.append({"tool": tool_name, "type": case["type"], "ok": False,
                            "error": str(e), "query": case["query"]})

        time.sleep(0.5)
        print()

    # ── Summary ──────────────────────────────────────────────────
    pct = 100 * passed // total if total else 0
    print(f"{'='*65}")
    print(f"RESULTS — {model.upper()}")
    print(f"  Total:    {total}")
    print(f"  ✅ Pass:  {passed}  ({pct}%)")
    print(f"  ❌ Fail:  {failed}")
    print(f"  ⚠️  Error: {errored}")

    # By type
    for ttype in ["read", "write", "generic"]:
        type_cases = [r for r in results if r.get("type") == ttype]
        type_pass  = sum(1 for r in type_cases if r.get("ok"))
        if type_cases:
            print(f"  {ttype.upper():8} {type_pass}/{len(type_cases)}")

    print(f"{'='*65}")

    # ── Failed details ───────────────────────────────────────────
    failed_results = [r for r in results if not r.get("ok")]
    if failed_results:
        print(f"\n{'='*65}")
        print("FAILED TOOLS:")
        for r in failed_results:
            print(f"  ❌ {r['tool']}: {r.get('issues') or r.get('error','?')}")

    # ── Save log ─────────────────────────────────────────────────
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model,
            "timestamp": timestamp,
            "summary": {"total": total, "passed": passed, "failed": failed, "errored": errored},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results → {log_file}")

    return results


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAU2 Airline Tool Benchmark")
    parser.add_argument("--model",  default="anthropic", choices=["anthropic", "openai", "both"])
    parser.add_argument("--tool",   default=None, help=f"Run one tool. Options: {list(TEST_CASES.keys())}")
    parser.add_argument("--domain", default=None, choices=["read", "write", "generic"],
                        help="Filter by tool type")
    args = parser.parse_args()

    if args.model in ("anthropic", "both") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY"); sys.exit(1)
    if args.model in ("openai", "both") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY"); sys.exit(1)

    if args.model == "both":
        run_all("anthropic", args.tool, args.domain)
        print()
        run_all("openai",    args.tool, args.domain)
    else:
        run_all(args.model, args.tool, args.domain)
