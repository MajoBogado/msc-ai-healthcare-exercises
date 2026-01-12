# 🧪 Rule-Based Clinical Triage System

## 📑 Table of Contents
- [Assumptions](#-assumptions)
- [Objective](#-objective)
- [Use Case](#-use-case)
- [System Overview](#-system-overview)
- [Forward vs Backward Reasoning](#-forward-vs-backward-reasoning)
- [Why This Is a Simplified Expert System](#-why-this-is-a-simplified-expert-system)
- [How to Run the System](#-how-to-run-the-system)
- [Validation Module](#-validation-module)
- [The Key Principle of rule-based Expert Systems](#-the-key-principle-of-rule-based-expert-systems)
- [What This Exercise Demonstrates](#-what-this-exercise-demonstrates)
- [Notes](#-notes)

---

## 📌 Assumptions

This exercise focuses on the **implementation of a rule-based expert system**, not on knowledge acquisition.

Therefore:

1. The expert rules are assumed to be already defined  
2. Patient data (facts) are provided as input examples  
3. No interaction with real clinicians or patients is required  

This reflects common practice in **early-stage expert system development** and **educational prototypes**, where the goal is to study system architecture and reasoning rather than clinical knowledge elicitation.

---

## 🎯 Objective

Implement a **simple rule-based expert system** that performs **basic clinical triage** using predefined rules and patient facts.

The goal is to demonstrate:

- Knowledge representation using **IF–THEN rules**
- **Forward- and backward-chaining** inference strategies
- Clear separation between:
  - facts (working memory)
  - rules (knowledge base)
  - inference logic (inference engine)

---

## 🏥 Use Case

A simplified **primary-care triage assistant** that classifies patients into one of three categories:

- `EMERGENCY`
- `SEE_DOCTOR`
- `HOME_CARE`

This system is intended **for educational purposes only** and does not perform diagnosis or provide medical advice.

---

## 🧠 System Overview

The system is composed of the following components:

- **Knowledge base** (`knowledge_base.py`)  
  Contains expert knowledge encoded as IF–THEN rules.

- **Facts / cases** (`cases.py`)  
  Represent patient data (symptoms and vital signs).

- **Inference engine** (`inference.py`)  
  Applies rules to facts using different reasoning strategies.

- **Entry point / interface** (`main.py`)  
  Executes the system and prints the reasoning process.

- **Validation module** (`validator.py`)  
  Verifies system behavior against expected outcomes.

---

## 🔍 Forward vs Backward Reasoning

### Forward chaining (data-driven)

- Starts from the **facts**
- Evaluates which rules are satisfied
- Applies a matching rule to reach a decision

**Mental model:**  
This approach is typical of **triage, alerts, and monitoring systems**.

---

### Backward chaining (goal-driven)

- Starts from a **hypothesis** (e.g. “Is this an emergency?”)
- Identifies rules that could justify that hypothesis
- Checks whether the facts satisfy those rules
- Confirms or rejects the hypothesis

**Mental model:**  
This approach is common in **diagnostic reasoning** and **explanation-oriented systems**.

---

## ⚠️ Why This Is a Simplified Expert System

This implementation is intentionally simplified.

### What it includes

- Deterministic IF–THEN rules
- Explicit working memory (facts)
- Forward and backward inference strategies
- Basic explanation of reasoning paths
- A validation mechanism using predefined cases

### What it does not include

- Knowledge acquisition from real experts
- Dynamic rule generation or learning
- Multi-step inference with new facts being generated
- Probabilistic or uncertain reasoning
- Integration with real clinical systems or EHRs

These simplifications keep the focus on **core expert system concepts**, making the system easier to understand and reason about.

---

## ▶️ How to Run the System

From the module directory:

```bash
python3 main.py

```
By default, **both forward and backward reasoning** are executed for each patient case.

## Optional parameters

Run **only forward chaining**:

```bash
python3 main.py --mode forward
```

Run only backward chaining:

```bash
python3 main.py --mode backward
```
---
## ✅ Validation Module

The validation module checks whether the system produces the **expected triage decision** for a set of predefined validation cases.

### Purpose

- Acts as a **verification database**, as described in expert system theory
- Ensures rule behavior is consistent and correct
- Allows forward and backward reasoning to be validated independently

### How to run validation

```bash
python3 validator.py
```

The validator reports pass/fail results for each case and summarizes overall performance.

## ✅ The key principle of rule-based expert systems

In a rule-based expert system, new knowledge can be added by introducing new rules without changing the inference mechanism.

This holds as long as:

- new rules follow the same IF–THEN structure
- inference strategy remains appropriate (forward/backward)

To add a new rule:
1. Use the JSON file located in the module directory. You can edit the file to test different outcomes.
2. From the module directory run:
```bash
python3 main.py --rules-file extra_rules.json
```

You will recognize the new rule by the ID. When the rules are loaded, it will appear as U1, then next will become U2, etc. The ids are being autogenerated in the backend.

## ✅ What This Exercise Demonstrates

By completing this exercise, you demonstrate understanding of:

- Rule-based expert system architecture
- Knowledge representation using IF–THEN rules
- Forward- and backward-chaining inference
- Explanation of reasoning paths
- Verification and validation of expert systems
- The role of expert systems in healthcare decision support

---

## 📌 Notes

This project is designed as a **conceptual and architectural exercise** aligned with introductory material on expert systems in healthcare AI.  
It prioritizes **clarity, interpretability, and correctness** over clinical completeness or performance.
