# main.py
import argparse

from cases import PATIENT_CASES
from knowledge_base import RULES as BASE_RULES
from inference import infer_forward, infer_backward
from rules_loader import load_rules_from_json


def print_block(title: str, lines: list[str]) -> None:
    print(title)
    print("-" * len(title))
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-Based Clinical Triage System"
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "backward"],
        help="Inference strategy to use (default: run both)",
    )
    parser.add_argument(
        "--rules-file",
        help="Path to JSON file containing additional rules to load",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Start with base knowledge base
    rules = list(BASE_RULES)

    # Optionally extend knowledge base with user-defined rules
    if args.rules_file:
        extra_rules = load_rules_from_json(args.rules_file, existing_rules=rules)
        rules.extend(extra_rules)

    run_forward = args.mode in (None, "forward")
    run_backward = args.mode in (None, "backward")

    print("Rule-Based Clinical Triage System")
    print("=" * 32)
    print(f"Mode: {args.mode if args.mode else 'both'}")
    print(f"Rules loaded: {len(rules)}")

    for case in PATIENT_CASES:
        case_id = case["id"]
        facts = case["facts"]

        print(f"\n\nCASE: {case_id}")
        print("=" * (6 + len(case_id)))

        if run_forward:
            _, _, f_lines = infer_forward(facts, rules)
            print_block("FORWARD CHAINING", f_lines)

        if run_forward and run_backward:
            print()

        if run_backward:
            _, _, b_lines = infer_backward(facts, rules)
            print_block("BACKWARD CHAINING", b_lines)


if __name__ == "__main__":
    main()
