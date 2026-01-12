# inference.py
from typing import Any, Dict, List, Optional, Tuple

Facts = Dict[str, Any]
Rule = Dict[str, Any]


def infer_forward(
    facts: Facts, rules: List[Rule]
) -> Tuple[str, Optional[Rule], List[str]]:
    """
    Forward chaining explanation format:
    Facts -> Rule applied -> Decision
    Returns:
      decision, applied_rule (or None), explanation_lines
    """
    lines: List[str] = []
    lines.append("Facts (working memory):")
    for k, v in facts.items():
        lines.append(f"  - {k}: {v}")

    lines.append("\nRules evaluation (in order):")
    for rule in rules:
        if rule["condition"](facts):
            lines.append(f"  ✅ Applied {rule['id']}: IF {rule['if_text']} THEN {rule['then_text']}")
            lines.append(f"\nDecision: {rule['action']}")
            return rule["action"], rule, lines
        else:
            lines.append(f"  ❌ Skipped {rule['id']}: IF {rule['if_text']} THEN {rule['then_text']}")

    lines.append("\nDecision: NO_DECISION (no rules matched)")
    return "NO_DECISION", None, lines


def infer_backward(
    facts: Facts, rules: List[Rule], goals: Optional[List[str]] = None
) -> Tuple[str, Optional[Rule], List[str]]:
    """
    Backward chaining explanation format:
    Hypothesis -> rule(s) that could prove it -> facts -> decision
    Returns:
      decision, proving_rule (or None), explanation_lines
    """
    if goals is None:
        goals = ["EMERGENCY", "SEE_DOCTOR", "HOME_CARE"]

    lines: List[str] = []

    # Print facts once (listing facts when a rule is checked/proven);
    facts_block = ["Facts available:"]
    for k, v in facts.items():
        facts_block.append(f"  - {k}: {v}")

    for goal in goals:
        lines.append(f"Hypothesis: triage={goal}")

        candidate_rules = [r for r in rules if r["action"] == goal]
        if not candidate_rules:
            lines.append("  (No rules can conclude this hypothesis)\n")
            continue

        lines.append("  Rules that could prove this hypothesis:")
        for r in candidate_rules:
            lines.append(f"    - {r['id']}: IF {r['if_text']} THEN {r['then_text']}")

        # Try to prove by facts
        proven_rule: Optional[Rule] = None
        for r in candidate_rules:
            lines.append(f"  Checking {r['id']} against facts...")
            if r["condition"](facts):
                proven_rule = r
                break

        if proven_rule:
            lines.append(f"  ✅ Hypothesis proven by {proven_rule['id']}")
            lines.append("  Facts used / available at proof time:")
            lines.extend([f"    {line}" for line in facts_block[1:]])  # indent facts
            lines.append(f"\nDecision: {goal}")
            return goal, proven_rule, lines

        lines.append("  ❌ Hypothesis not proven (no candidate rule matched the facts)\n")

    lines.append("Decision: NO_DECISION (no hypothesis could be proven)")
    return "NO_DECISION", None, lines
