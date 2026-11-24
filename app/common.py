"""
Shared code for every page of the website: loading the model and the reference data once.
Streamlit keeps both in memory (st.cache_resource), so pages open quickly after the first load.
"""
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
import booking   # noqa: E402  (helpers for the website, src/booking.py)
import pipeline  # noqa: E402  (cleaning, features and training, src/pipeline.py)

MODEL_PATH = ROOT / "models" / "model.joblib"
FIGURES = ROOT / "figures"
GITHUB = "https://github.com/ixanm10/hotel-cancellation-prediction"


@st.cache_resource(show_spinner="Training the model for the first time (about 1 minute)...")
def load_bundle():
    """The trained model. If the file is not there yet (first start on Streamlit Cloud), train it."""
    if not MODEL_PATH.exists():
        pipeline.run(quick=True)
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Preparing the data...")
def load_reference():
    """Clean data, the same 80/20 split as the notebook, and the model's predictions on the test set."""
    from sklearn.model_selection import train_test_split

    b = load_bundle()
    df, _ = pipeline.add_features(pipeline.clean(pipeline.load_raw()), top_countries=b["top_countries"])
    X, y = df[b["num"] + b["cat"]], df["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=pipeline.RANDOM_STATE)
    typical = booking.typical_booking(df.loc[X_train.index])
    return {
        "df": df,
        "y_test": y_test.to_numpy(),
        "proba_test": b["model"].predict_proba(X_test)[:, 1],
        "typical": typical,
        "cancel_rate": float(y.mean()),
    }


def sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.caption("CDS 4003 – Group 14")
    st.sidebar.markdown(f"[GitHub repository]({GITHUB})")
