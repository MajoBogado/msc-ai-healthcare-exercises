RULES = [
    {
        "id": "R1",
        "if_text": "chest_pain=True AND shortness_of_breath=True",
        "then_text": "triage=EMERGENCY",
        "description": "Chest pain with shortness of breath",
        "condition": lambda f: f["chest_pain"] and f["shortness_of_breath"],
        "action": "EMERGENCY",
    },
    {
        "id": "R2",
        "if_text": "oxygen_saturation < 92",
        "then_text": "triage=EMERGENCY",
        "description": "Low oxygen saturation",
        "condition": lambda f: f["oxygen_saturation"] < 92,
        "action": "EMERGENCY",
    },
    {
        "id": "R3",
        "if_text": "fever=True AND heart_rate > 100",
        "then_text": "triage=SEE_DOCTOR",
        "description": "Fever with high heart rate",
        "condition": lambda f: f["fever"] and f["heart_rate"] > 100,
        "action": "SEE_DOCTOR",
    },
    {
        "id": "R4",
        "if_text": "fever=False AND chest_pain=False",
        "then_text": "triage=HOME_CARE",
        "description": "No fever and no chest pain",
        "condition": lambda f: (not f["fever"]) and (not f["chest_pain"]),
        "action": "HOME_CARE",
    },
]
