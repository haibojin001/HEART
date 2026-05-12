"""
test_retail_all.py — TAU2 Retail 全工具 Benchmark 测试脚本

测试 16 个 retail 工具的 schema 能否被 LLM 正确识别和调用。
所有测试数据均来自真实 db.json。

用法：
  export ANTHROPIC_API_KEY=sk-ant-...
  python test_retail_all.py                         # 测全部 16 个工具
  python test_retail_all.py --model openai          # 使用 OpenAI
  python test_retail_all.py --model both            # 两个都测
  python test_retail_all.py --domain read           # 只测 READ 工具
  python test_retail_all.py --domain write          # 只测 WRITE 工具
  python test_retail_all.py --tool cancel_pending_order  # 测单个
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from data_model import RetailDB
from tools import RetailTools

DB_PATH     = Path(__file__).parent / "db.json"
POLICY_PATH = Path(__file__).parent / "policy.md"


def load_tools() -> RetailTools:
    db = RetailDB.load(str(DB_PATH))
    return RetailTools(db)


def get_policy() -> str:
    return POLICY_PATH.read_text() if POLICY_PATH.exists() else \
        "You are a retail customer service agent."


# ══════════════════════════════════════════════════════════════════
# 真实数据说明（来自 db.json）
# ──────────────────────────────────────────────────────────────────
# 主测试用户: raj_sanchez_2970
#   email:    raj.sanchez2046@example.com
#   zip:      92147
#   payments: credit_card_3362387 (mastercard 2130)
#             gift_card_2259499   (balance: $30)
#
# 待处理订单 #W4566809 (pending, paid by credit_card_3362387, $826.70)
#   item 1569829406: Indoor Security Camera $320.55
#     → alt: 8470360507 $291.31 (2K, 130°, Ethernet)
#   item 5826601160: Air Purifier $506.15
#     → alt: 8302289002 $547.55 (large, HEPA, night mode)
#
# 已送达订单 #W7736708 (delivered, paid by credit_card_3362387, $980.87)
#   item 3104857380: Headphones $377.97 (on-ear, wireless, red)
#     → alt: 3374679624 $370.53 (over-ear, wired, black)
#   item 9692325258: Luggage Set $528.63
#
# 产品:
#   Indoor Security Camera product_id: (looked up at runtime)
#   T-Shirt product_id: 9523456873
#   item 9612497925: T-Shirt blue M $50.88
# ══════════════════════════════════════════════════════════════════

# ── Pre-built tool result stubs ───────────────────────────────────

_USER_RAJ = json.dumps({
    "user_id": "raj_sanchez_2970",
    "name": {"first_name": "Raj", "last_name": "Sanchez"},
    "email": "raj.sanchez2046@example.com",
    "address": {
        "address1": "557 Sunset Drive", "address2": "Suite 454",
        "city": "San Diego", "state": "CA", "country": "USA", "zip": "92147"
    },
    "payment_methods": {
        "credit_card_3362387": {
            "source": "credit_card", "id": "credit_card_3362387",
            "brand": "mastercard", "last_four": "2130"
        },
        "gift_card_2259499": {
            "source": "gift_card", "id": "gift_card_2259499", "balance": 30.0
        }
    },
    "orders": ["#W4566809", "#W7736708", "#W1067251"]
})

_ORDER_PENDING = json.dumps({
    "order_id": "#W4566809",
    "user_id": "raj_sanchez_2970",
    "status": "pending",
    "address": {
        "address1": "557 Sunset Drive", "address2": "Suite 454",
        "city": "San Diego", "state": "CA", "country": "USA", "zip": "92147"
    },
    "items": [
        {
            "item_id": "1569829406", "product_id": "7038065355",
            "name": "Indoor Security Camera", "price": 320.55,
            "options": {"resolution": "1080p", "field of view": "160 degrees", "connectivity": "Ethernet"}
        },
        {
            "item_id": "5826601160", "product_id": "2941248747",
            "name": "Air Purifier", "price": 506.15,
            "options": {"room size": "medium", "filter type": "carbon", "features": "night mode"}
        }
    ],
    "payment_history": [
        {"transaction_type": "payment", "amount": 826.70, "payment_method_id": "credit_card_3362387"}
    ],
    "fulfillments": [],
    "cancel_reason": None,
})

_ORDER_DELIVERED = json.dumps({
    "order_id": "#W7736708",
    "user_id": "raj_sanchez_2970",
    "status": "delivered",
    "address": {
        "address1": "557 Sunset Drive", "address2": "Suite 454",
        "city": "San Diego", "state": "CA", "country": "USA", "zip": "92147"
    },
    "items": [
        {
            "item_id": "3104857380", "product_id": "9775965776",
            "name": "Headphones", "price": 377.97,
            "options": {"type": "on-ear", "connectivity": "wireless", "color": "red"}
        },
        {
            "item_id": "9692325258", "product_id": "9016994464",
            "name": "Luggage Set", "price": 528.63,
            "options": {"piece count": "3-piece", "color": "black", "material": "softshell"}
        }
    ],
    "payment_history": [
        {"transaction_type": "payment", "amount": 980.87, "payment_method_id": "credit_card_3362387"}
    ],
    "fulfillments": [{"tracking_id": ["TRK123456"], "item_ids": ["3104857380", "9692325258"]}],
    "cancel_reason": None,
})


# ══════════════════════════════════════════════════════════════════
# 16 个工具测试用例
# ══════════════════════════════════════════════════════════════════

TEST_CASES = {

    # ── READ / GENERIC (直接触发) ─────────────────────────────────

    "find_user_id_by_email": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. Can you find my account?"},
        ],
        "expected_tool": "find_user_id_by_email",
        "expected_args": {"email": "raj.sanchez2046@example.com"},
        "note": "Authentication step 1: lookup by email",
    },

    "find_user_id_by_name_zip": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "I can't remember my email. My name is Raj Sanchez and my zip code is 92147."},
        ],
        "expected_tool": "find_user_id_by_name_zip",
        "expected_args": {"first_name": "Raj", "last_name": "Sanchez", "zip": "92147"},
        "note": "Authentication step 1 alternative: lookup by name + zip",
    },

    "get_user_details": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. Can you show me my account details?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001",
                 "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content":
             "I found your account. Let me pull up your full details."},
            {"role": "user", "content": "Yes, show me my full account."},
        ],
        "expected_tool": "get_user_details",
        "expected_args": {"user_id": "raj_sanchez_2970"},
        "note": "After email auth, get full user profile",
    },

    "get_order_details": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. What's the status of my order #W4566809?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_user_details",
                 "input": {"user_id": "raj_sanchez_2970"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _USER_RAJ},
            ]},
            {"role": "assistant", "content":
             "I've verified your identity. Let me look up order #W4566809."},
            {"role": "user", "content": "Yes please."},
        ],
        "expected_tool": "get_order_details",
        "expected_args": {"order_id": "#W4566809"},
        "note": "Lookup pending order after auth",
    },

    "get_product_details": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "Can you show me the details for product 9523456873?"},
        ],
        "expected_tool": "get_product_details",
        "expected_args": {"product_id": "9523456873"},
        "note": "Direct product lookup (T-Shirt)",
    },

    "get_item_details": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "Can you show me the details for item 9612497925?"},
        ],
        "expected_tool": "get_item_details",
        "expected_args": {"item_id": "9612497925"},
        "note": "Direct item/variant lookup (T-Shirt blue M)",
    },

    "list_all_product_types": {
        "type": "read",
        "messages": [
            {"role": "user", "content":
             "What types of products do you sell? Can I see a list?"},
        ],
        "expected_tool": "list_all_product_types",
        "expected_args": {},
        "note": "List all 50 product types",
    },

    "calculate": {
        "type": "generic",
        "messages": [
            {"role": "user", "content":
             "Calculate 320.55 + 506.15 for me."},
        ],
        "expected_tool": "calculate",
        "expected_args": {"expression": "320.55 + 506.15"},
        "note": "Math expression",
    },

    "transfer_to_human_agents": {
        "type": "generic",
        "messages": [
            {"role": "user", "content":
             "I need to speak to a human agent. My issue cannot be resolved by automated support."},
        ],
        "expected_tool": "transfer_to_human_agents",
        "expected_args": {},
        "note": "Explicit request for human, summary auto-generated",
    },

    # ── WRITE (多轮历史，已到确认阶段) ────────────────────────────

    "cancel_pending_order": {
        "type": "write",
        # #W4566809: pending, paid credit_card_3362387 $826.70
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to cancel order #W4566809. "
             "I ordered it by mistake."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W4566809"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_PENDING},
            ]},
            {"role": "assistant", "content":
             "I can cancel order #W4566809. Here are the details:\n"
             "- Order: #W4566809 (pending)\n"
             "- Items: Indoor Security Camera + Air Purifier\n"
             "- Reason: ordered by mistake\n"
             "- Refund: $826.70 to mastercard ending 2130 (5-7 business days)\n"
             "Shall I proceed?"},
            {"role": "user", "content": "Yes, please cancel it."},
        ],
        "expected_tool": "cancel_pending_order",
        "expected_args": {"order_id": "#W4566809", "reason": "ordered by mistake"},
        "note": "Pending order, credit card refund, reason=ordered by mistake",
    },

    "modify_pending_order_address": {
        "type": "write",
        # Change shipping address of #W4566809
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I need to change the shipping address "
             "for order #W4566809 to 100 New Street, Apt 5, Los Angeles, CA, USA, 90001."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W4566809"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_PENDING},
            ]},
            {"role": "assistant", "content":
             "I'll update the shipping address for order #W4566809:\n"
             "- New address: 100 New Street, Apt 5, Los Angeles, CA, USA 90001\n"
             "Shall I proceed?"},
            {"role": "user", "content": "Yes, update the address."},
        ],
        "expected_tool": "modify_pending_order_address",
        "expected_args": {
            "order_id": "#W4566809",
            "address1": "100 New Street",
            "city": "Los Angeles",
            "state": "CA",
            "country": "USA",
            "zip": "90001",
        },
        "note": "Pending order, change shipping address",
    },

    "modify_pending_order_payment": {
        "type": "write",
        # Change payment from credit_card to gift_card (balance $30 < $826.70, but let's use a different scenario)
        # Actually gift card balance $30 is not enough. Let's say user wants to switch to gift_card for a smaller portion
        # Better: use a different order with smaller amount, or find order paid with gift card to switch to credit card
        # Use #W4566809 $826.70, switch FROM credit_card TO gift_card — will fail at execution (balance too low)
        # But we just need LLM to CALL the tool correctly — execution result doesn't matter for schema test
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to change the payment method "
             "for order #W4566809 from credit card to gift card gift_card_2259499."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W4566809"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_PENDING},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_003", "name": "get_user_details",
                 "input": {"user_id": "raj_sanchez_2970"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_003", "content": _USER_RAJ},
            ]},
            {"role": "assistant", "content":
             "I'll change the payment method for order #W4566809:\n"
             "- From: mastercard ending 2130\n"
             "- To: gift_card_2259499 (balance: $30.00)\n"
             "- Order total: $826.70\n"
             "Shall I proceed?"},
            {"role": "user", "content": "Yes, please change the payment method."},
        ],
        "expected_tool": "modify_pending_order_payment",
        "expected_args": {"order_id": "#W4566809", "payment_method_id": "gift_card_2259499"},
        "note": "Pending order, change payment method (schema test only)",
    },

    "modify_pending_order_items": {
        "type": "write",
        # Change Indoor Security Camera from 1569829406 to 8470360507 (2K version, $291.31 vs $320.55)
        # Price diff: -$29.24 (refund to credit_card_3362387)
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to change the Indoor Security Camera "
             "in order #W4566809 to the 2K version (item 8470360507). "
             "Use credit_card_3362387 for any price difference."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W4566809"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_PENDING},
            ]},
            {"role": "assistant", "content":
             "I'll modify order #W4566809:\n"
             "- Change item 1569829406 (Indoor Security Camera 1080p $320.55)\n"
             "  → item 8470360507 (Indoor Security Camera 2K $291.31)\n"
             "- Price difference: -$29.24 (refund to mastercard ending 2130)\n"
             "Is this the only item you want to change? Shall I proceed?"},
            {"role": "user", "content": "Yes, that's the only change. Please proceed."},
        ],
        "expected_tool": "modify_pending_order_items",
        "expected_args": {
            "order_id": "#W4566809",
            "item_ids": ["1569829406"],
            "new_item_ids": ["8470360507"],
            "payment_method_id": "credit_card_3362387",
        },
        "note": "Pending order, swap item to different variant, refund $29.24",
    },

    "modify_user_address": {
        "type": "write",
        # Change user's default address
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to update my default address "
             "to 200 Ocean Ave, Unit 3B, Santa Monica, CA, USA, 90401."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_user_details",
                 "input": {"user_id": "raj_sanchez_2970"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _USER_RAJ},
            ]},
            {"role": "assistant", "content":
             "I'll update your default address to:\n"
             "200 Ocean Ave, Unit 3B, Santa Monica, CA, USA 90401\n"
             "Shall I proceed?"},
            {"role": "user", "content": "Yes, update my address."},
        ],
        "expected_tool": "modify_user_address",
        "expected_args": {
            "user_id": "raj_sanchez_2970",
            "address1": "200 Ocean Ave",
            "city": "Santa Monica",
            "state": "CA",
            "country": "USA",
            "zip": "90401",
        },
        "note": "Update user's default shipping address",
    },

    "return_delivered_order_items": {
        "type": "write",
        # Return Headphones (3104857380) from delivered order #W7736708
        # Refund to original credit_card_3362387
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to return the Headphones "
             "from my delivered order #W7736708. Refund to my credit card credit_card_3362387."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W7736708"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_DELIVERED},
            ]},
            {"role": "assistant", "content":
             "I'll process a return for order #W7736708:\n"
             "- Returning: Headphones (item 3104857380) — $377.97\n"
             "- Refund to: mastercard ending 2130 (5-7 business days)\n"
             "Is this the only item you want to return? Shall I proceed?"},
            {"role": "user", "content": "Yes, just the headphones. Please proceed."},
        ],
        "expected_tool": "return_delivered_order_items",
        "expected_args": {
            "order_id": "#W7736708",
            "item_ids": ["3104857380"],
            "payment_method_id": "credit_card_3362387",
        },
        "note": "Delivered order, return 1 item, refund to original credit card",
    },

    "exchange_delivered_order_items": {
        "type": "write",
        # Exchange Headphones 3104857380 → 3374679624 (over-ear, $370.53, price diff -$7.44)
        # Refund diff to credit_card_3362387
        "messages": [
            {"role": "user", "content":
             "My email is raj.sanchez2046@example.com. I want to exchange the Headphones "
             "in delivered order #W7736708. I want the over-ear wired black version (item 3374679624). "
             "Use credit_card_3362387 for any price difference."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_001", "name": "find_user_id_by_email",
                 "input": {"email": "raj.sanchez2046@example.com"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": '"raj_sanchez_2970"'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_002", "name": "get_order_details",
                 "input": {"order_id": "#W7736708"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_002", "content": _ORDER_DELIVERED},
            ]},
            {"role": "assistant", "content":
             "I'll process the exchange for order #W7736708:\n"
             "- Exchange: item 3104857380 (Headphones on-ear wireless red $377.97)\n"
             "  → item 3374679624 (Headphones over-ear wired black $370.53)\n"
             "- Price difference: -$7.44 (refund to mastercard ending 2130)\n"
             "Is this the only item you want to exchange? Shall I proceed?"},
            {"role": "user", "content": "Yes, just those headphones. Please proceed."},
        ],
        "expected_tool": "exchange_delivered_order_items",
        "expected_args": {
            "order_id": "#W7736708",
            "item_ids": ["3104857380"],
            "new_item_ids": ["3374679624"],
            "payment_method_id": "credit_card_3362387",
        },
        "note": "Delivered order, exchange 1 item, refund $7.44",
    },

}


# ══════════════════════════════════════════════════════════════════
# Schema generation
# ══════════════════════════════════════════════════════════════════

def get_schemas_anthropic(tools: RetailTools) -> list[dict]:
    schemas = []
    for name, tool in tools.get_tools().items():
        fn = tool.openai_schema["function"]
        schemas.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return schemas


def get_schemas_openai(tools: RetailTools) -> list[dict]:
    return [tool.openai_schema for tool in tools.get_tools().values()]


# ══════════════════════════════════════════════════════════════════
# Tool execution
# ══════════════════════════════════════════════════════════════════

def execute_tool(tools: RetailTools, tool_name: str, tool_input: dict) -> str:
    from datetime import date, datetime
    from pydantic import BaseModel

    def serialize(obj):
        if isinstance(obj, BaseModel):         return obj.model_dump()
        elif isinstance(obj, (datetime, date)): return obj.isoformat()
        elif isinstance(obj, list):             return [serialize(i) for i in obj]
        elif isinstance(obj, tuple):            return [serialize(i) for i in obj]
        elif isinstance(obj, dict):             return {k: serialize(v) for k, v in obj.items()}
        return obj

    try:
        result = tools.use_tool(tool_name, **tool_input)
        return json.dumps(serialize(result), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# API runners
# ══════════════════════════════════════════════════════════════════

def run_anthropic(messages: list, tools: RetailTools) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=get_policy(),
        tools=get_schemas_anthropic(tools),
        messages=messages,
    )
    for block in response.content:
        if block.type == "tool_use":
            return {"called": True, "name": block.name, "input": block.input, "id": block.id}
    return {"called": False, "stop_reason": response.stop_reason}


def run_openai(messages: list, tools: RetailTools) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    oai_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            tool_uses    = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_uses:
                oai_messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])}
                    } for b in tool_uses]
                })
            elif tool_results:
                for tr in tool_results:
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"],
                    })
        else:
            oai_messages.append({"role": msg["role"], "content": content})

    oai_messages.insert(0, {"role": "system", "content": get_policy()})

    response = client.chat.completions.create(
        model="gpt-4o",
        tools=get_schemas_openai(tools),
        tool_choice="auto",
        messages=oai_messages,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {"called": True, "name": tc.function.name,
                "input": json.loads(tc.function.arguments), "id": tc.id}
    return {"called": False, "finish_reason": response.choices[0].finish_reason}


# ══════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════

def verify(result: dict, expected_tool: str, expected_args: dict) -> tuple[bool, list[str]]:
    if not result.get("called"):
        return False, [f"No tool called (stop: {result.get('stop_reason') or result.get('finish_reason','?')})"]
    issues = []
    if result["name"] != expected_tool:
        return False, [f"Wrong tool: got '{result['name']}', expected '{expected_tool}'"]
    for key, val in expected_args.items():
        if key not in result["input"]:
            issues.append(f"Missing arg '{key}'")
        elif val is not None and result["input"][key] != val:
            issues.append(f"Arg '{key}': got {result['input'][key]!r}, expected {val!r}")
    return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════

def run_all(model: str = "anthropic", tool_filter=None, domain_filter=None):
    tools = load_tools()
    runner = run_anthropic if model == "anthropic" else run_openai

    cases = {
        k: v for k, v in TEST_CASES.items()
        if (tool_filter is None or k == tool_filter)
        and (domain_filter is None or v["type"] == domain_filter)
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = f"retail_results_{model}_{timestamp}.json"

    print(f"\n{'='*65}")
    print(f"Retail Tool Benchmark — {model.upper()}  ({len(cases)} tools)")
    if tool_filter:   print(f"Tool:   {tool_filter}")
    if domain_filter: print(f"Domain: {domain_filter}")
    print(f"{'='*65}\n")

    results = []
    total = passed = failed = errored = 0

    for i, (name, case) in enumerate(cases.items(), 1):
        total += 1
        turns = len([m for m in case["messages"] if m["role"] == "user"])
        print(f"[{i:02d}/{len(cases)}] {case['type'].upper():7} | {name}  ({turns}-turn)")
        print(f"  note: {case['note']}")

        try:
            result = runner(case["messages"], tools)
            ok, issues = verify(result, case["expected_tool"], case["expected_args"])

            if ok:
                passed += 1
                print(f"  ✅ Called: {result['name']}")
                preview = dict(list(result["input"].items())[:3])
                more = " ..." if len(result["input"]) > 3 else ""
                print(f"  input: {json.dumps(preview, ensure_ascii=False)}{more}")
            else:
                failed += 1
                print(f"  ❌ FAIL")
                for issue in issues:
                    print(f"     → {issue}")
                if result.get("called"):
                    print(f"     actual: {result['name']}({json.dumps(result.get('input',{}))[:120]})")

            results.append({
                "tool": name, "type": case["type"],
                "ok": ok, "issues": issues, "result": result,
            })

        except Exception as e:
            errored += 1
            print(f"  ⚠️  Error: {e}")
            results.append({"tool": name, "type": case["type"], "ok": False, "error": str(e)})

        time.sleep(0.4)
        print()

    # Summary
    pct = 100 * passed // total if total else 0
    print(f"{'='*65}")
    print(f"RESULTS — {model.upper()}")
    print(f"  Total:    {total}")
    print(f"  ✅ Pass:  {passed}  ({pct}%)")
    print(f"  ❌ Fail:  {failed}")
    print(f"  ⚠️  Error: {errored}")
    for ttype in ["read", "write", "generic"]:
        grp = [r for r in results if r.get("type") == ttype]
        ok  = sum(1 for r in grp if r.get("ok"))
        if grp: print(f"  {ttype.upper():8} {ok}/{len(grp)}")
    print(f"{'='*65}")

    failed_list = [r for r in results if not r.get("ok")]
    if failed_list:
        print(f"\nFailed: {[r['tool'] for r in failed_list]}")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model, "timestamp": timestamp,
            "summary": {"total": total, "passed": passed, "failed": failed, "errored": errored},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {log_file}")
    return results


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Tool Benchmark")
    parser.add_argument("--model",  default="anthropic", choices=["anthropic","openai","both"])
    parser.add_argument("--tool",   default=None, help=f"Single tool: {list(TEST_CASES.keys())}")
    parser.add_argument("--domain", default=None, choices=["read","write","generic"])
    args = parser.parse_args()

    if args.model in ("anthropic","both") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY"); sys.exit(1)
    if args.model in ("openai","both") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY"); sys.exit(1)

    if args.model == "both":
        run_all("anthropic", args.tool, args.domain)
        print()
        run_all("openai",    args.tool, args.domain)
    else:
        run_all(args.model, args.tool, args.domain)
