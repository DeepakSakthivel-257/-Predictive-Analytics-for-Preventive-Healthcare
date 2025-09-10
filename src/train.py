import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .utils import read_heart_csv, train_val_test_split, FEATURES, TARGET

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed.csv")
    parser.add_argument("--model", required=True, help="Path to save best model pipeline (joblib)")
    args = parser.parse_args()

    df = read_heart_csv(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    # Candidate pipelines
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    rf_pipe = Pipeline([("clf", RandomForestClassifier(random_state=42))])

    # Parameter grids
    lr_grid = {"clf__C": [0.1, 1.0, 10.0]}
    rf_grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10]}

    # Grid search (ROC-AUC)
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    rf_search = GridSearchCV(rf_pipe, rf_grid, cv=5, scoring="roc_auc", n_jobs=-1)

    lr_search.fit(X_train, y_train)
    rf_search.fit(X_train, y_train)

    # Validation comparison
    def evaluate(pipe):
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_val)[:, 1]
        else:
            proba = pipe.decision_function(X_val)
        y_pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, proba)
        return acc, f1, auc

    lr_acc, lr_f1, lr_auc = evaluate(lr_search.best_estimator_)
    rf_acc, rf_f1, rf_auc = evaluate(rf_search.best_estimator_)

    best = lr_search if lr_auc >= rf_auc else rf_search
    best_name = "LogisticRegression" if best is lr_search else "RandomForest"

    # Final fit on train+val
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    best.best_estimator_.fit(X_tv, y_tv)

    # Held-out test evaluation
    pipe = best.best_estimator_
    if hasattr(pipe, "predict_proba"):
        proba_test = pipe.predict_proba(X_test)[:, 1]
    else:
        proba_test = pipe.decision_function(X_test)
    y_pred_test = (proba_test >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, proba_test)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": FEATURES, "target": TARGET, "best_model_name": best_name}, args.model)

    print(f"[train] Best model: {best_name}")
    print(f"[train] Test metrics: ACC={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")
    print(f"[train] Saved pipeline to {args.model}")

if __name__ == "__main__":
    main()
