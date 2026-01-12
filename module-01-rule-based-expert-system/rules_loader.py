# rules_loader.py
import json
import re
from typing import Any, Dict, List, Callable, Optional

Facts = Dict[str, Any]
Rule = Dict[str, Any]

_ALLOWED_OPS = {"==", "!=", "<", "<=", ">", ">="}
_USER_PREFIX = "U"


def _make_condition(all_of: List[Dict[str, Any]]) -> Callable[[Facts], bool]:
    def cond(facts: Facts) -> bool:
        for c in all_of:
            field = c["field"]
            op = c["op"]
            value = c["value"]

            if op not in _ALLOWED_OPS:
                raise ValueError(f"Unsupported operator: {op}")

            if field not in facts:
                return False

            fval = facts[field]

            if op == "==":
                ok = fval == value
            elif op == "!=":
                ok = fval != value
            elif op == "<":
                ok = fval < value
            elif op == "<=":
                ok = fval <= value
            elif op == ">":
                ok = fval > value
            elif op == ">=":
                ok = fval >= value
            else:
                ok = False

            if not ok:
                return False
        return True

    return cond


def _next_user_id(existing_rules: List[Rule]) -> str:
    """
    Generates the next ID like U1, U2, U3...
    Looks at existing rule IDs starting with 'U'.
    """
    pattern = re.compile(rf"^{_USER_PREFIX}(\d+)$")
    max_n = 0
    for r in existing_rules:
        rid = r.get("id", "")
        m = pattern.match(rid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{_USER_PREFIX}{max_n + 1}"


def load_rules_from_json(path: str, existing_rules: List[Rule]) -> List[Rule]:
    """
    Load user rules from JSON and auto-assign IDs if missing.

    JSON schema (id is optional):
    [
      {
        "description": "...",
        "if_text": "...",
        "then_text": "...",
        "action": "EMERGENCY",
        "all_of": [{"field":"oxygen_saturation","op":"<","value":88}]
      }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Rules JSON must be a list of rule objects")

    new_rules: List[Rule] = []
    # We generate IDs sequentially, considering base + already-generated ones
    temp_existing = list(existing_rules)

    for r in data:
        all_of = r.get("all_of", [])
        if not isinstance(all_of, list) or len(all_of) == 0:
            raise ValueError("Each rule must contain a non-empty 'all_of' list")

        rid = r.get("id")
        if not rid:
            rid = _next_user_id(temp_existing)

        rule: Rule = {
            "id": rid,
            "description": r.get("description", ""),
            "if_text": r.get("if_text", ""),
            "then_text": r.get("then_text", f"triage={r.get('action', '')}"),
            "action": r["action"],
            "condition": _make_condition(all_of),
        }

        new_rules.append(rule)
        temp_existing.append(rule)  # so the next generated id increments properly

    return new_rules
