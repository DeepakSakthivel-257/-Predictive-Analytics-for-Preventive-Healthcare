import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from .utils import read_heart_csv, train_val_test_split, FEATURES, TARGET

def save_confusion_matrix(y_true, y_pred, out_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_roc(y_true, y_proba, out_path):
    plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_pr(y_true, y_proba, out_path):
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_feature_importance(pipe, feature_names, out_path):
    plt.figure()
    if hasattr(pipe[-1], "feature_importances_"):
        importances = pipe[-1].feature_importances_
        order = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[order])
        plt.xticks(range(len(importances)), [feature_names[i] for i in order], rotation=45, ha="right")
        plt.title("Feature Importances (Tree-based)")
    elif hasattr(pipe[-1], "coef_"):
        coefs = pipe[-1].coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1]
        plt.bar(range(len(coefs)), coefs[order])
        plt.xticks(range(len(coefs)), [feature_names[i] for i in order], rotation=45, ha="right")
        plt.title("Feature Coefficients (LogReg)")
    else:
        plt.text(0.5, 0.5, "No native feature importance available", ha="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed.csv")
    parser.add_argument("--model", required=True, help="Path to trained .joblib pipeline")
    parser.add_argument("--plots", required=True, help="Directory to save plots")
    args = parser.parse_args()

    df = read_heart_csv(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    features = bundle["features"]

    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipe.decision_function(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    outdir = Path(args.plots)
    outdir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(y_test, y_pred, outdir / "confusion_matrix.png")
    save_roc(y_test, y_proba, outdir / "roc_curve.png")
    save_pr(y_test, y_proba, outdir / "precision_recall_curve.png")
    save_feature_importance(pipe, features, outdir / "feature_importance.png")

    print(f"[evaluate] Saved plots to: {outdir}")

if __name__ == "__main__":
    main()
