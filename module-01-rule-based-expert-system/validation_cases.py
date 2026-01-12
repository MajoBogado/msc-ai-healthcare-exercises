VALIDATION_CASES = [
    {
        "id": "val_1_emergency_breathing",
        "facts": {
            "fever": False,
            "chest_pain": True,
            "shortness_of_breath": True,
            "heart_rate": 95,
            "oxygen_saturation": 96,
        },
        "expected": "EMERGENCY",
    },
    {
        "id": "val_2_emergency_low_o2",
        "facts": {
            "fever": True,
            "chest_pain": False,
            "shortness_of_breath": False,
            "heart_rate": 90,
            "oxygen_saturation": 90,
        },
        "expected": "EMERGENCY",
    },
    {
        "id": "val_3_see_doctor_fever_tachy",
        "facts": {
            "fever": True,
            "chest_pain": False,
            "shortness_of_breath": False,
            "heart_rate": 110,
            "oxygen_saturation": 98,
        },
        "expected": "SEE_DOCTOR",
    },
]
