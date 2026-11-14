"""
Hotel Booking Cancellation Prediction - end-to-end pipeline
CDS 4003 Group 3

Runs the same CRISP-DM steps as notebooks/Hotel_Cancellation_Colab.ipynb:
  1. Load raw data            (data/hotel_bookings.csv)
  2. Clean + prepare          -> data/hotel_bookings_clean.csv
  3. Build features
  4. Train + compare models
  5. Evaluate on hold-out test set -> outputs/metrics.json, outputs/test_results.csv, outputs/figures/
  6. Save the best model      -> models/model.joblib (used by app/app.py)

Usage:
  python src/pipeline.py            # full run with hyperparameter tuning (like the notebook)
  python src/pipeline.py --quick    # fast run without tuning (used by GitHub Actions)
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no screen needed (runs on GitHub servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "hotel_bookings.csv"
CLEAN_PATH = ROOT / "data" / "hotel_bookings_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42
COST_FN, COST_FP = 180, 20  # euros - business assumptions from the report

MONTH_MAP = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
             "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

NUM_FEATURES = ["lead_time", "arrival_month_num", "arrival_date_week_number", "arrival_date_day_of_month",
                "arrival_dow", "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
                "adults", "children", "babies", "total_guests", "has_kids", "is_repeated_guest",
                "previous_cancellations", "previous_bookings_not_canceled", "prev_cancel_rate",
                "booking_changes", "days_in_waiting_list", "adr", "total_revenue", "weekend_ratio",
                "required_car_parking_spaces", "total_of_special_requests", "is_summer"]
CAT_FEATURES = ["hotel", "meal", "country_grp", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type"]


# ---------------------------------------------------------------- 1-2. Load and clean
def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.drop(columns=["company"])                       # 94% missing
    df["children"] = df["children"].fillna(df["children"].mode()[0])
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df = df.drop_duplicates()
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)]
    df["meal"] = df["meal"].replace("Undefined", "SC")
    df = df[(df["market_segment"] != "Undefined") & (df["distribution_channel"] != "Undefined")]
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month_num"],
                                             day=df["arrival_date_day_of_month"]))
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    df["children"] = df["children"].astype(int)
    df["agent"] = df["agent"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 3. Features
def add_features(df: pd.DataFrame, top_countries=None):
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_kids"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["total_revenue"] = df["adr"] * df["total_nights"]
    df["weekend_ratio"] = df["stays_in_weekend_nights"] / df["total_nights"]
    prev_total = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_rate"] = (df["previous_cancellations"] / prev_total.replace(0, np.nan)).fillna(0)
    df["arrival_dow"] = df["arrival_date"].dt.dayofweek
    df["is_summer"] = df["arrival_month_num"].isin([6, 7, 8]).astype(int)
    df["lead_time_band"] = pd.cut(df["lead_time"], [-1, 7, 30, 90, 180, 365, 800],
                                  labels=["0-7", "8-30", "31-90", "91-180", "181-365", "365+"])
    if top_countries is None:
        top_countries = df["country"].value_counts().head(15).index.tolist()
    df["country_grp"] = np.where(df["country"].isin(top_countries), df["country"], "Other")
    df["room_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    return df, top_countries


# ---------------------------------------------------------------- 4. Models
def build_models():
    pre_linear = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)])
    pre_tree = ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES)])
    return {
        "Baseline (majority class)": Pipeline([("pre", pre_linear), ("model", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([("pre", pre_linear), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Naive Bayes": Pipeline([("pre", pre_linear), ("model", GaussianNB())]),
        "Decision Tree": Pipeline([("pre", pre_tree), ("model", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("pre", pre_tree), ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosted Trees": Pipeline([("pre", pre_tree), ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
    }


def tune(models, X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    s = GridSearchCV(models["Decision Tree"], {"model__max_depth": [6, 10, 14, 20],
                     "model__min_samples_leaf": [1, 10, 50]}, scoring="f1", cv=cv, n_jobs=-1)
    tuned["Decision Tree (tuned)"] = s.fit(X_train, y_train).best_estimator_
    print("  Decision Tree best:", s.best_params_)
    s = RandomizedSearchCV(models["Random Forest"], {"model__n_estimators": [200, 300], "model__max_depth": [None, 20, 30],
                           "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.5]},
                           n_iter=5, scoring="f1", cv=cv, n_jobs=1, random_state=RANDOM_STATE)
    tuned["Random Forest (tuned)"] = s.fit(X_train, y_train).best_estimator_
    print("  Random Forest best:", s.best_params_)
    s = RandomizedSearchCV(models["Gradient Boosted Trees"], {"model__learning_rate": [0.05, 0.1, 0.2],
                           "model__max_iter": [200, 400, 600], "model__max_leaf_nodes": [31, 63, 127],
                           "model__min_samples_leaf": [20, 50], "model__l2_regularization": [0.0, 1.0]},
                           n_iter=10, scoring="f1", cv=cv, n_jobs=-1, random_state=RANDOM_STATE)
    tuned["Gradient Boosted Trees (tuned)"] = s.fit(X_train, y_train).best_estimator_
    print("  Gradient Boosted Trees best:", s.best_params_)
    return tuned


# ---------------------------------------------------------------- 5. Evaluate
def evaluate(model, X_, y_, threshold=0.5):
    proba = model.predict_proba(X_)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_, pred), "precision": precision_score(y_, pred, zero_division=0),
            "recall": recall_score(y_, pred), "specificity": tn / (tn + fp), "f1": f1_score(y_, pred),
            "AUC": roc_auc_score(y_, proba), "false_negatives": int(fn), "false_positives": int(fp),
            "total_cost_eur": int(fn * COST_FN + fp * COST_FP)}


def save_figures(df, final_models, best_name, X_test, y_test):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Lead time
    lead = df.groupby("lead_time_band", observed=True)["is_canceled"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lead.index.astype(str), lead, color="#2b6cb0")
    ax.axhline(df["is_canceled"].mean() * 100, color="#dd6b20", ls="--", label="overall rate")
    ax.set_xlabel("Lead time (days)"); ax.set_ylabel("Cancellation rate (%)"); ax.legend()
    ax.set_title("Cancellation rate rises with lead time")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_lead_time.png", dpi=150); plt.close(fig)
    # Confusion matrix
    best = final_models[best_name]
    cm = confusion_matrix(y_test, best.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not cancelled", "Cancelled"]).plot(cmap="Blues", values_format=",")
    disp.ax_.set_title(f"Confusion matrix - {best_name}")
    disp.figure_.savefig(FIG_DIR / "fig3_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close(disp.figure_)
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in final_models.items():
        if name.startswith("Baseline"):
            continue
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
    ax.set_title("ROC curves on the test set")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_roc.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def run(quick=False):
    print("1. Loading", RAW_PATH.name)
    raw = load_raw()
    print("   rows:", len(raw), "| columns:", raw.shape[1])

    print("2. Cleaning")
    df = clean(raw)
    df, top_countries = add_features(df)
    df.to_csv(CLEAN_PATH, index=False)
    print("   clean rows:", len(df), "| cancellation rate: %.1f%%" % (df["is_canceled"].mean() * 100))

    print("3. Splitting train / test (80 / 20, stratified)")
    X, y = df[NUM_FEATURES + CAT_FEATURES], df["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print("4. Training models" + (" (quick mode, no tuning)" if quick else ""))
    models = build_models()
    final_models = {}
    for name, m in models.items():
        if not quick and name in ("Decision Tree", "Random Forest", "Gradient Boosted Trees"):
            continue
        final_models[name] = m.fit(X_train, y_train)
        print("   trained:", name)
    if not quick:
        final_models.update(tune(models, X_train, y_train))

    print("5. Evaluating on the hold-out test set")
    results = pd.DataFrame({n: evaluate(m, X_test, y_test) for n, m in final_models.items()}).T
    results = results.sort_values("f1", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(OUT_DIR / "test_results.csv")
    print(results[["accuracy", "precision", "recall", "f1", "AUC"]].round(3).to_string())

    best_name = results["f1"].idxmax()
    best = results.loc[best_name]
    metrics = {"best_model": best_name, **{k: float(best[k]) for k in ["accuracy", "precision", "recall", "f1", "AUC"]},
               "rows_train": int(len(X_train)), "rows_test": int(len(X_test)), "quick_mode": quick}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_figures(df, final_models, best_name, X_test, y_test)

    print("6. Saving best model:", best_name)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_models[best_name], "model_name": best_name, "num": NUM_FEATURES,
                 "cat": CAT_FEATURES, "top_countries": list(top_countries)}, MODEL_PATH, compress=3)
    print("   saved to", MODEL_PATH.relative_to(ROOT))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotel cancellation prediction pipeline")
    parser.add_argument("--quick", action="store_true", help="skip hyperparameter tuning (fast)")
    run(quick=parser.parse_args().quick)
