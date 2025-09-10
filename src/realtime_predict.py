import argparse
import joblib
import numpy as np
from .utils import FEATURES
from .recommendations import get_recommendations

def main():
    parser = argparse.ArgumentParser(description="Realtime prediction CLI for Heart Disease risk")
    parser.add_argument("--model", required=True, help="Path to trained .joblib pipeline")
    # Add one argument per feature
    for feat in FEATURES:
        parser.add_argument(f"--{feat}", type=float, required=True)
    args = parser.parse_args()

    X = np.array([[getattr(args, f) for f in FEATURES]], dtype=float)
    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]

    # Compute prediction
    proba = pipe.predict_proba(X)[0, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)[0]
    y_pred = int(proba >= 0.5)

    # Print prediction
    print("=== Heart Disease Risk Prediction ===")
    print(f"Input features order: {FEATURES}")
    print(f"Predicted class: {y_pred} (1 = disease, 0 = no disease)")
    print(f"Risk probability: {proba:.3f}")

    # Get treatment advice
    advice = get_recommendations(y_pred)
    print("\n--- Recommendations ---")
    print(advice["message"])
    if advice["treatments"]:
        print("\nTreatments / Medications:")
        for t in advice["treatments"]:
            print(f"- {t}")
    print("\nLifestyle / Cautions:")
    for l in advice["lifestyle_cautions"]:
        print(f"- {l}")

if __name__ == "__main__":
    main()
