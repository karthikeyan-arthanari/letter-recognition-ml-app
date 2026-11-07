"""
train_models.py
---------------
Trains the five classifiers required by ML Assignment 2 on the UCI "Letter Recognition"
dataset, evaluates them on a held-out test set and saves everything the Streamlit app needs.

Run from the project root:

    python model/train_models.py

Outputs
    data/train_data.csv            training split (used only for the app's fallback retraining)
    test_data.csv                  held-out test split (upload this file in the Streamlit app)
    model/*.joblib                 one saved pipeline per model
    model/metrics_summary.csv      comparison table (Accuracy, AUC, Precision, Recall, F1, MCC)
    model/metadata.json            feature names, class labels, chosen hyper-parameters
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "letterdata.csv"
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "letter"
SEED = 42
TEST_SIZE = 0.20

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_models(knn_k=5, svm_c=10.0):
    """Return the five (untrained) scikit-learn pipelines.
    Scale-sensitive models (LR, kNN, SVM) get a StandardScaler; tree and Gaussian NB do not need one."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=SEED))]),
        "kNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=knn_k))]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB())]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=svm_c, gamma="scale", probability=True, random_state=SEED))]),
    }


def compute_metrics(y_true, y_pred, y_proba, classes):
    """The six metrics required by the assignment.
    The problem is multi-class (26 letters), so precision / recall / F1 are macro-averaged and
    AUC is the one-vs-rest macro average. MCC is scikit-learn's multi-class generalisation."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    present = [c for c in classes if c in set(y_true)]
    auc = np.nan
    if len(present) >= 2:
        cols = [int(np.where(classes == c)[0][0]) for c in present]
        p = y_proba[:, cols]
        p = p / p.sum(axis=1, keepdims=True)          # renormalise if some classes are absent
        auc = roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=present)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def load_clean_dataset():
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates().reset_index(drop=True)   # exact duplicate rows would leak between train/test
    return df


def tune_hyperparameters(X_train, y_train):
    """Small, honest 3-fold CV search on the TRAINING split only (the test split is never touched)."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    knn_scores = {}
    for k in (1, 3, 5, 7, 9, 11):
        m = Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k))])
        knn_scores[k] = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    svm_scores = {}
    for c in (1.0, 10.0, 30.0):
        m = Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", C=c, gamma="scale"))])
        svm_scores[c] = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    best_k = max(knn_scores, key=knn_scores.get)
    best_c = max(svm_scores, key=svm_scores.get)
    return {"knn_k": int(best_k), "svm_c": float(best_c),
            "cv_knn_accuracy": {str(k): round(v, 4) for k, v in knn_scores.items()},
            "cv_svm_accuracy": {str(k): round(v, 4) for k, v in svm_scores.items()}}


def main():
    df = load_clean_dataset()
    print(f"Dataset after removing duplicates: {df.shape[0]} rows x {df.shape[1] - 1} features + target")

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    features = [c for c in df.columns if c != TARGET]
    X_train, y_train = train_df[features], train_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    hp = tune_hyperparameters(X_train, y_train)
    print("Chosen hyper-parameters:", {k: hp[k] for k in ("knn_k", "svm_c")})

    rows = {}
    for name, model in build_models(hp["knn_k"], hp["svm_c"]).items():
        t0 = time.time()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        rows[name] = compute_metrics(y_test, pred, proba, model.classes_)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"{name:24s} acc={rows[name]['Accuracy']:.4f}  ({time.time() - t0:.1f}s)")

    summary = pd.DataFrame(rows).T.round(4)
    summary.index.name = "ML Model Name"
    summary.to_csv(MODEL_DIR / "metrics_summary.csv")
    meta = {"target": TARGET, "features": features, "classes": [str(c) for c in model.classes_],
            "seed": SEED, "test_size": TEST_SIZE, "hyperparameters": hp,
            "n_rows_after_dedup": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df))}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()
