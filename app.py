"""
Streamlit app - ML Assignment 2 (Letter Recognition, 26 classes)

Features required by the assignment
  1. Dataset upload option (CSV, test data only)
  2. Model selection dropdown (5 models)
  3. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  4. Confusion matrix / classification report

Run locally:  streamlit run app.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.train_models import MODEL_FILES, build_models, compute_metrics

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
META = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES, TARGET, CLASSES = META["features"], META["target"], META["classes"]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Letter Recognition - Model Explorer", page_icon="🔤", layout="wide")


# ------------------------------------------------------------------ model loading
@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    """Load the saved pipelines. If the pickles cannot be read by the scikit-learn version installed
    on the server (versions differ between machines), retrain quickly from data/train_data.csv instead."""
    models, retrained = {}, False
    for name, fname in MODEL_FILES.items():
        try:
            models[name] = joblib.load(MODEL_DIR / fname)
        except Exception:
            retrained = True
            break
    if retrained:
        train = pd.read_csv(BASE / "data" / "train_data.csv")
        hp = META["hyperparameters"]
        models = build_models(hp["knn_k"], hp["svm_c"])
        for m in models.values():
            m.fit(train[FEATURES], train[TARGET])
    return models, retrained


@st.cache_data
def load_reference_table():
    return pd.read_csv(MODEL_DIR / "metrics_summary.csv", index_col=0)


models, was_retrained = load_models()
reference = load_reference_table()

# ------------------------------------------------------------------ header / sidebar
st.title("🔤 Letter Recognition - Classification Model Explorer")
st.caption("UCI Letter Recognition dataset: 16 integer features describing a distorted capital letter, "
           "26 classes (A-Z). Upload test data, pick a model and inspect how it performs.")

with st.sidebar:
    st.header("1. Choose a model")
    model_name = st.selectbox("Classification model", list(MODEL_FILES.keys()), index=4)
    st.header("2. Provide test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help=f"Must contain the 16 feature columns. Add the '{TARGET}' column to get metrics.")
    use_sample = st.checkbox("Use the bundled test_data.csv", value=uploaded is None,
                             disabled=uploaded is not None)
    st.markdown("---")
    st.caption(f"Models trained on {META['n_train']:,} rows (80 %), tested on {META['n_test']:,} rows (20 %), "
               f"seed {META['seed']}.")
    if was_retrained:
        st.info("Saved models were incompatible with this server's scikit-learn - models were re-trained at start-up.")

# ------------------------------------------------------------------ data
data = None
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file as CSV: {exc}")
elif use_sample:
    data = pd.read_csv(BASE / "test_data.csv")

tab_eval, tab_compare, tab_about = st.tabs(["📊 Evaluate selected model", "🏁 Compare all models", "ℹ️ About the data"])

with tab_eval:
    if data is None:
        st.info("Upload a CSV in the sidebar (or tick the bundled sample) to begin.")
    else:
        data = data.copy()
        data.columns = [c.strip() for c in data.columns]
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"The file is missing required feature columns: {', '.join(missing)}")
        else:
            X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
            bad = X.isna().any(axis=1)
            if bad.any():
                st.warning(f"{int(bad.sum())} row(s) with non-numeric / empty feature values were dropped.")
            keep = ~bad
            X = X[keep]
            has_labels = TARGET in data.columns
            y = data.loc[keep, TARGET].astype(str).str.strip().str.upper() if has_labels else None

            model = models[model_name]
            pred = model.predict(X)
            proba = model.predict_proba(X)

            st.subheader(f"{model_name} - results on {len(X):,} uploaded rows")
            if has_labels:
                unknown = sorted(set(y) - set(CLASSES))
                if unknown:
                    st.warning(f"Labels not seen in training were found ({', '.join(unknown)}); "
                               "they are counted as errors but excluded from AUC.")
                m = compute_metrics(y, pred, proba, model.classes_)
                cols = st.columns(6)
                for col, key in zip(cols, METRIC_ORDER):
                    val = m[key]
                    col.metric(key, "n/a" if pd.isna(val) else f"{val:.4f}")

                left, right = st.columns([3, 2])
                with left:
                    st.markdown("**Confusion matrix** (rows = actual letter, columns = predicted letter)")
                    labels = sorted(set(CLASSES) | set(y) | set(pred))
                    cm = confusion_matrix(y, pred, labels=labels)
                    normalise = st.toggle("Show row-normalised values (recall per letter)", value=False)
                    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
                    fig, ax = plt.subplots(figsize=(9, 7))
                    sns.heatmap(shown, ax=ax, cmap="Blues", xticklabels=labels, yticklabels=labels,
                                fmt=".2f" if normalise else "d", annot=False, cbar=True, linewidths=.2, linecolor="#e8e8e8")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                    st.pyplot(fig, clear_figure=True)
                with right:
                    st.markdown("**Classification report**")
                    rep = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T
                    rep = rep.rename(columns={"f1-score": "f1"}).round(3)
                    rep["support"] = rep["support"].astype(int)
                    st.dataframe(rep, height=520, width="stretch")

                err = pd.DataFrame({"actual": y.values, "predicted": pred})
                err = err[err.actual != err.predicted]
                st.markdown(f"**Most frequent mistakes** ({len(err):,} wrong predictions)")
                if len(err):
                    top = (err.value_counts().reset_index(name="count").head(8)
                           .rename(columns={"actual": "actual letter", "predicted": "predicted as"}))
                    st.dataframe(top, hide_index=True, width="content")
                else:
                    st.success("No mistakes on this file.")
            else:
                st.info(f"The file has no '{TARGET}' column, so only predictions are shown (metrics need true labels).")

            out = X.copy()
            out["predicted_letter"] = pred
            out["confidence"] = proba.max(axis=1).round(4)
            with st.expander("Predictions table / download"):
                st.dataframe(out.head(200), width="stretch")
                st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(),
                                   file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                                   mime="text/csv")

with tab_compare:
    st.subheader("Metrics of all five models on the held-out 20 % test split")
    st.caption("Computed once by model/train_models.py - macro-averaged Precision / Recall / F1, one-vs-rest macro AUC, multi-class MCC.")
    st.dataframe(reference.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    chart = reference[METRIC_ORDER].T
    st.bar_chart(chart)

with tab_about:
    st.markdown(f"""
**Target:** `{TARGET}` - one of 26 capital letters.  **Features ({len(FEATURES)}):** {', '.join(f'`{f}`' for f in FEATURES)}.

All features are integers between 0 and 15 (statistical moments and edge counts of the letter image).
Models: Logistic Regression, Decision Tree, k-Nearest Neighbours (k = {META['hyperparameters']['knn_k']}),
Gaussian Naive Bayes and an RBF Support Vector Machine (C = {META['hyperparameters']['svm_c']:g}).
""")
    st.dataframe(pd.read_csv(BASE / "test_data.csv").head(10), width="stretch")
