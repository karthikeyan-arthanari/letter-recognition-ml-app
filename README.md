# Letter Recognition – Classification Models and Streamlit App

**ML Assignment 2 · M.Tech Data Science & Engineering · BITS Pilani WILP**
Karthikeyan · BITS ID 2025NS04128

* GitHub repository: `<PASTE YOUR GITHUB REPO LINK HERE>`
* Live Streamlit app: `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

---

## a. Problem statement

A printed capital letter (A–Z) is shown as a small, slightly distorted black-and-white raster image. Instead of working with the pixels, a
set of 16 numeric statistics (position, size, pixel counts, moments and edge counts of the letter) is computed for each image.
The task is to **predict which of the 26 letters** each image shows from those 16 numbers.

This is a **26-class classification problem**. The goal of the assignment is to implement five different classifiers on the same data,
compare them with six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and expose the trained models through a Streamlit web
app where a test CSV can be uploaded and evaluated.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Letter Recognition Data Set |
| Source | UCI Machine Learning Repository (D. J. Slate & P. W. Frey, 1991) |
| Original size | 20,000 rows × 16 features + 1 target |
| Used after cleaning | **18,668 rows** (1,332 exact duplicate rows removed so that identical rows cannot appear in both train and test) |
| Features | **16**, all integers in the range 0–15 |
| Target | `letter` – 26 classes (A … Z), almost perfectly balanced (about 105–156 rows per letter in the test split) |
| Missing values | none |
| Train / test split | 80 % / 20 % stratified, `random_state = 42` → 14,934 train rows, 3,734 test rows |

Feature columns: `xbox, ybox` (box position), `width, height` (box size), `onpix` (number of “on” pixels), `xbar, ybar` (mean x / y of on pixels),
`x2bar, y2bar` (mean variance in x / y), `xybar` (mean x·y correlation), `x2ybar, xy2bar` (mean of x²y and xy²), `xedge, yedge`
(mean edge count left→right / bottom→top) and `xedgey, yedgex` (correlation of edge counts with y / x).

The data was generated from 20 different fonts that were randomly distorted, so many rows of the same letter look similar to each other,
which matters for how the models behave (see the observations below).

## c. GitHub repository

`<PASTE YOUR GITHUB REPO LINK HERE>`

```
letter-recognition-ml-app/
├── app.py                 Streamlit application
├── requirements.txt       Python dependencies
├── README.md              this file
├── test_data.csv          held-out 20 % test split (upload in the app)
├── data/
│   ├── letterdata.csv     original dataset
│   └── train_data.csv     80 % training split (app fallback re-training)
└── model/
    ├── train_models.py    training + evaluation script
    ├── ml_assignment2_letter_recognition.ipynb   executed notebook
    ├── *.joblib           five saved model pipelines
    ├── metrics_summary.csv   comparison table
    └── metadata.json      features, classes, hyper-parameters
```

To run locally:

```bash
pip install -r requirements.txt
python model/train_models.py      # optional: re-creates splits, models and metrics
streamlit run app.py
```

## d. Models used

All five models were trained on the same training split and evaluated on the same held-out test split.
Because the target has 26 classes, Precision / Recall / F1 are **macro-averaged**, AUC is the **one-vs-rest macro average**,
and MCC is the **multi-class** version.
Logistic Regression, kNN and SVM are wrapped in a pipeline with a `StandardScaler`.
The settings of kNN (k) and SVM (C) were chosen by 3-fold cross-validation on the training data only.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7633 | 0.9778 | 0.7663 | 0.7620 | 0.7626 | 0.7538 |
| Decision Tree | 0.8677 | 0.9309 | 0.8689 | 0.8672 | 0.8677 | 0.8624 |
| kNN (k = 1) | 0.9440 | 0.9708 | 0.9451 | 0.9439 | 0.9443 | 0.9418 |
| Naive Bayes (Gaussian) | 0.6256 | 0.9554 | 0.6405 | 0.6246 | 0.6228 | 0.6114 |
| Support Vector Machine (RBF, C = 30) | **0.9692** | **0.9993** | **0.9702** | **0.9688** | **0.9693** | **0.9680** |

(Bold = best value in the column. The same numbers are in `model/metrics_summary.csv` and in the “Compare all models” tab of the app.)

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest of the “sensible” models on hard predictions (accuracy 0.763, MCC 0.754). A single linear boundary per letter cannot separate shapes such as G/C/Q or P/F, so its mistakes are spread widely (hardest letters: G, H, S with F1 ≈ 0.57). Interestingly its AUC (0.978) is high: the probabilities rank the right letter near the top even when the top-1 guess is wrong, and, unlike the 1-NN or the unpruned tree, it returns soft probabilities instead of near 0/1 votes. |
| Decision Tree | Good, cheap and fully interpretable baseline (accuracy 0.868). It beats the two linear/probabilistic models but is clearly worse than kNN and SVM because an unpruned tree over-fits the training fonts and makes axis-aligned splits. It has the lowest AUC (0.931) since leaves give almost 0/1 probabilities, so there is little ranking information. Typical errors: P→F, R→K, Q→O. |
| kNN | Very strong (accuracy 0.944, MCC 0.942). CV picked k = 1, i.e. the model simply finds the most similar stored letter – which works because the same 20 fonts occur in both training and test data. Weak spots: H/K and I/J. Because k = 1 returns hard 0/1 probabilities, its AUC (0.971) is lower than Logistic Regression's although its accuracy is almost 18 points higher. It is also the slowest at prediction time because it stores all training rows. |
| Naive Bayes | Lowest scores on every hard-prediction metric (accuracy 0.626, MCC 0.611). Gaussian Naive Bayes assumes the 16 features are independent and normally distributed given the letter; here the features are integer-valued, skewed and strongly correlated, so the assumption is badly violated (worst letters: S, H, X; most common error Q→O). It is, however, the fastest to train (≈0.1 s) and its AUC (0.955) is still reasonable, i.e. the ranking of letters is better than its top-1 choice. |
| Support Vector Machine | Best model on all six metrics (accuracy 0.969, AUC 0.9993, MCC 0.968). The RBF kernel gives a smooth non-linear boundary that generalises across fonts better than the single nearest neighbour; larger C (30) was preferred by cross-validation. Only 115 of the 3,734 test rows are wrong and no letter has F1 below 0.93 (hardest: R, B, H). Cost: the slowest to train (about 10 s here, including probability calibration) and the largest model file. |

### Overall – winner for this dataset

**Support Vector Machine (RBF kernel)** is the best model for this dataset: it has the highest value in all six metrics, ahead of kNN by
about 2.5 points of accuracy / MCC. The ranking by MCC is **SVM (0.968) > kNN (0.942) > Decision Tree (0.862) > Logistic Regression (0.754) >
Naive Bayes (0.611)**. Models that can draw flexible non-linear boundaries (SVM, kNN) are clearly better than the linear and
independence-based ones on this problem. kNN is a close, simpler alternative if training time matters.

## Streamlit app features

1. **Dataset upload** – CSV upload in the sidebar (test data only; must contain the 16 feature columns, and a `letter` column if metrics are wanted). A checkbox can load the bundled `test_data.csv`.
2. **Model selection** – dropdown with the five models.
3. **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1 and MCC displayed for the uploaded data.
4. **Confusion matrix and classification report** – heat-map (with optional row-normalisation) and a per-letter report, plus the most frequent mistakes and downloadable predictions.

Extras: a “Compare all models” tab, column validation with clear error messages, and automatic re-training from `data/train_data.csv`
if the saved models cannot be loaded by the server’s scikit-learn version.

## Notes on the method

* Duplicates were removed **before** the split to avoid train/test leakage.
* The test split was never used for model selection; k and C were tuned with 3-fold CV on the training split.
* All results are reproducible with `python model/train_models.py` (fixed seed 42) and match the executed notebook.
