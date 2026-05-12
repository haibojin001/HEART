from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from ..schemas import ToolParameter, ToolSchema
from ..toolface import ToolFace


_BANK = {
    "users": {
        "u_001": {
            "name": "Alex Chen",
            "email": "alex@example.com",
            "phone": "+1-217-555-0143",
            "cards": ["c_4321", "c_9988"],
        },
        "u_002": {
            "name": "Maya Patel",
            "email": "maya@example.com",
            "phone": "+1-217-555-0177",
            "cards": ["c_1234"],
        },
    },
    "cards": {
        "c_4321": {"last4": "4321", "type": "debit", "status": "active",  "owner": "u_001"},
        "c_9988": {"last4": "9988", "type": "credit","status": "active",  "owner": "u_001"},
        "c_1234": {"last4": "1234", "type": "debit", "status": "active",  "owner": "u_002"},
    },
    "transactions": {
        "u_001": [
            {"id": "tx_a", "card": "c_4321", "merchant": "Whole Foods",  "amount":  43.12, "ts": 1714512000},
            {"id": "tx_b", "card": "c_4321", "merchant": "Amazon",       "amount": 199.99, "ts": 1714612000},
            {"id": "tx_c", "card": "c_9988", "merchant": "Delta Airlines","amount": 612.00, "ts": 1714712000},
        ],
        "u_002": [
            {"id": "tx_d", "card": "c_1234", "merchant": "Target",       "amount":  88.50, "ts": 1714812000},
        ],
    },
}


def get_user_profile(user_id: str) -> Dict[str, Any]:
    if user_id not in _BANK["users"]:
        raise ValueError(f"unknown user_id: {user_id}")
    u = _BANK["users"][user_id]
    return {"user_id": user_id, **u}


def list_user_cards(user_id: str) -> List[Dict[str, Any]]:
    if user_id not in _BANK["users"]:
        raise ValueError(f"unknown user_id: {user_id}")
    cards = []
    for cid in _BANK["users"][user_id]["cards"]:
        c = _BANK["cards"][cid]
        cards.append({"card_id": cid, "last4": c["last4"], "type": c["type"], "status": c["status"]})
    return cards


def freeze_debit_card(card_id: str, op: str = "lock") -> Dict[str, Any]:
    if op not in ("lock", "unlock"):
        raise ValueError(f"op must be lock|unlock, got {op!r}")
    if card_id == "auto":
        affected = []
        for cid, c in _BANK["cards"].items():
            if c["type"] == "debit" and c["status"] == "active":
                c["status"] = "frozen" if op == "lock" else "active"
                affected.append(cid)
        return {
            "output": f"Auto-froze {len(affected)} debit card(s)",
            "data": {"lock_id": f"LX-{uuid.uuid4().hex[:4].upper()}",
                     "affected_cards": affected,
                     "timestamp": time.strftime("%H:%M:%S")},
        }
    if card_id not in _BANK["cards"]:
        raise ValueError(f"unknown card_id: {card_id}")
    c = _BANK["cards"][card_id]
    c["status"] = "frozen" if op == "lock" else "active"
    return {
        "output": f"Card [**{c['last4']}] is {c['status'].upper()}",
        "data": {"lock_id": f"LX-{uuid.uuid4().hex[:4].upper()}",
                 "card_id": card_id,
                 "timestamp": time.strftime("%H:%M:%S")},
    }


def list_recent_transactions(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if user_id not in _BANK["users"]:
        raise ValueError(f"unknown user_id: {user_id}")
    txs = sorted(_BANK["transactions"].get(user_id, []), key=lambda t: t["ts"], reverse=True)
    return txs[:limit]


def report_card_stolen(card_id: str, contact_method: str = "email") -> Dict[str, Any]:
    if card_id not in _BANK["cards"]:
        raise ValueError(f"unknown card_id: {card_id}")
    if contact_method not in ("email", "sms", "call"):
        raise ValueError(f"contact_method must be one of email|sms|call")
    case_id = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    return {
        "case_id": case_id,
        "card_id": card_id,
        "status": "reported_stolen",
        "follow_up": contact_method,
        "next_steps": ["card replacement initiated", "shipping in 3-5 business days"],
    }


def register(tf: ToolFace) -> None:
    tf.register(
        ToolSchema(
            id="get_user_profile",
            name="Get user profile",
            description="Retrieve a user's profile (name, email, phone, registered cards).",
            category="finance",
            source="tau2",
            parameters=[
                ToolParameter("user_id", "string", required=True,
                              description="Bank-side user id, e.g., 'u_001'."),
            ],
            returns="Dict with user_id, name, email, phone, cards (list of card_ids).",
        ),
        get_user_profile,
    )
    tf.register(
        ToolSchema(
            id="list_user_cards",
            name="List user cards",
            description="List all cards (debit and credit) registered to a user.",
            category="finance",
            source="tau2",
            parameters=[
                ToolParameter("user_id", "string", required=True,
                              description="Bank-side user id."),
            ],
            returns="List of {card_id, last4, type, status}.",
        ),
        list_user_cards,
    )
    tf.register(
        ToolSchema(
            id="freeze_debit_card",
            name="Freeze debit card",
            description="Freeze (lock) or unfreeze (unlock) a debit card. "
                        "The canonical Figure-1 example.",
            category="finance",
            source="tau2",
            parameters=[
                ToolParameter("card_id", "string", required=True,
                              description="Either a concrete card_id (e.g., 'c_4321') "
                                          "or 'auto' to act on all debit cards of the user."),
                ToolParameter("op", "string", required=False, default="lock",
                              enum=["lock", "unlock"],
                              description="lock to freeze, unlock to reactivate."),
            ],
            returns="{output, data:{lock_id, timestamp, ...}}",
        ),
        freeze_debit_card,
    )
    tf.register(
        ToolSchema(
            id="list_recent_transactions",
            name="List recent transactions",
            description="List a user's most recent transactions, newest first.",
            category="finance",
            source="tau2",
            parameters=[
                ToolParameter("user_id", "string", required=True,
                              description="Bank-side user id."),
                ToolParameter("limit", "integer", required=False, default=10,
                              description="Max number of transactions to return."),
            ],
            returns="List of {id, card, merchant, amount, ts}.",
        ),
        list_recent_transactions,
    )
    tf.register(
        ToolSchema(
            id="report_card_stolen",
            name="Report card stolen",
            description="File a stolen-card report. Initiates replacement.",
            category="finance",
            source="tau2",
            parameters=[
                ToolParameter("card_id", "string", required=True,
                              description="The stolen card's id."),
                ToolParameter("contact_method", "string", required=False, default="email",
                              enum=["email", "sms", "call"],
                              description="How to contact the user for follow-up."),
            ],
            returns="{case_id, card_id, status, follow_up, next_steps}",
        ),
        report_card_stolen,
    )
