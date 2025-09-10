def get_recommendations(prediction):
    """
    Returns treatment and caution advice based on heart disease prediction.
    prediction: int (0 = no disease, 1 = disease)
    """
    if prediction == 1:
        advice = {
            "message": "Patient is at risk of heart disease.",
            "treatments": [
                "Consult a cardiologist immediately.",
                "Medications as prescribed by the doctor: Aspirin, Statins, Beta-blockers, ACE inhibitors.",
                "Regular heart checkups and ECG monitoring."
            ],
            "lifestyle_cautions": [
                "Reduce salt and saturated fat intake.",
                "Avoid smoking and alcohol.",
                "Exercise moderately (as advised by doctor).",
                "Maintain healthy weight and monitor blood pressure.",
                "Manage stress through meditation or therapy."
            ]
        }
    else:
        advice = {
            "message": "Patient has no heart disease detected.",
            "treatments": [],
            "lifestyle_cautions": [
                "Maintain a balanced diet.",
                "Exercise regularly.",
                "Avoid smoking and excessive alcohol.",
                "Monitor cholesterol and blood pressure periodically."
            ]
        }
    return advice
