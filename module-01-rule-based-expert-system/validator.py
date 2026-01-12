# validator.py
from knowledge_base import RULES
from validation_cases import VALIDATION_CASES
from inference import infer_forward, infer_backward


def _rule_label(rule) -> str:
    if not rule:
        return "None"
    rid = rule.get("id", "?")
    if_text = rule.get("if_text", rule.get("description", ""))
    then_text = rule.get("then_text", f"triage={rule.get('action', '')}")
    return f"{rid} (IF {if_text} THEN {then_text})"


def validate() -> None:
    total = len(VALIDATION_CASES)
    forward_ok = 0
    backward_ok = 0

    print("Validation runner")
    print("=" * 17)

    for case in VALIDATION_CASES:
        case_id = case["id"]
        facts = case["facts"]
        expected = case["expected"]

        # NEW: inference returns (decision, rule_dict_or_None, explanation_lines)
        f_decision, f_rule, _ = infer_forward(facts, RULES)
        b_decision, b_rule, _ = infer_backward(facts, RULES)

        f_pass = f_decision == expected
        b_pass = b_decision == expected

        forward_ok += int(f_pass)
        backward_ok += int(b_pass)

        print(f"\nCase: {case_id}")
        print(f"Expected: {expected}")

        print(f"Forward:  {f_decision}  {'✅' if f_pass else '❌'}")
        print(f"  Rule: {_rule_label(f_rule)}")

        print(f"Backward: {b_decision}  {'✅' if b_pass else '❌'}")
        print(f"  Rule: {_rule_label(b_rule)}")

    print("\n" + "-" * 40)
    print(f"Forward passed:  {forward_ok}/{total}")
    print(f"Backward passed: {backward_ok}/{total}")


if __name__ == "__main__":
    validate()
