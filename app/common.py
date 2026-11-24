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


# ---------------------------------------------------------------- look and feel
# "Old Library" palette
WALNUT, OXBLOOD, OLIVE, PAPER, PARCHMENT, VELVET = "#2E2522", "#5A302C", "#74613F", "#B99A70", "#E5D8C2", "#7A1020"

# Each page has its own gradient banner.
PAGE_COLORS = {
    "home": [VELVET, OXBLOOD],
    "predict": [OXBLOOD, WALNUT],
    "batch": [OLIVE, OXBLOOD],
    "insights": [WALNUT, OLIVE],
    "performance": [PAPER, OXBLOOD],
}
CARD_ACCENTS = [OXBLOOD, OLIVE, PAPER, WALNUT]  # top border of metric cards, left to right

CSS = """
<style>
/* sidebar: walnut-to-oxblood with parchment text */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #2E2522 0%%, #5A302C 100%%); }
[data-testid="stSidebar"] * { color: #E5D8C2 !important; }
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(229, 216, 194, 0.16) !important; border-radius: 8px; }

/* headings in walnut / oxblood */
h1, h2, h3 { color: #5A302C; }

/* metric boxes become cards with a coloured top */
[data-testid="stMetric"] { background: #FBF7EF; border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(46, 37, 34, 0.12); border-top: 5px solid #5A302C; }
[data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] { border-top-color: %s; }
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { border-top-color: %s; }
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { border-top-color: %s; }
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { border-top-color: %s; }

/* rounded buttons */
.stButton > button, .stDownloadButton > button { border-radius: 999px; font-weight: 600; }

/* page banner */
.banner { border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(46, 37, 34, 0.25); }
.banner h1 { color: #E5D8C2 !important; margin: 0; padding: 0; font-size: 2.1rem; font-family: Georgia, serif; }
.banner p { color: #E5D8C2; opacity: 0.92; margin: 6px 0 0 0; font-size: 1.05rem; }

/* risk result card */
.risk-card { border-radius: 16px; padding: 18px 22px; color: #F3ECDF; margin: 8px 0 14px 0;
    box-shadow: 0 4px 14px rgba(46, 37, 34, 0.2); }
.risk-card .level { font-size: 1.9rem; font-weight: 800; letter-spacing: 1px; font-family: Georgia, serif; }
.risk-card .prob { font-size: 1.1rem; opacity: 0.95; }
.risk-card .action { margin-top: 6px; font-size: 1.05rem; }
</style>
""" % tuple(CARD_ACCENTS)

RISK_COLORS = {"LOW": "linear-gradient(90deg, #5E6B3A, #74613F)",
               "MEDIUM": "linear-gradient(90deg, #A8782F, #B99A70)",
               "HIGH": "linear-gradient(90deg, #7A1020, #A3162A)"}
RISK_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}


def style():
    """Apply the website's custom styling (called once from app.py)."""
    st.markdown(CSS, unsafe_allow_html=True)


def banner(page, title, subtitle=""):
    """Gradient banner at the top of a page."""
    colors = ", ".join(PAGE_COLORS[page])
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="banner" style="background: linear-gradient(120deg, {colors});">'
                f"<h1>{title}</h1>{sub}</div>", unsafe_allow_html=True)


def risk_card(level, p, action):
    """Big coloured card with the risk level, probability and suggested action."""
    st.markdown(f'<div class="risk-card" style="background: {RISK_COLORS[level]};">'
                f'<div class="level">{RISK_ICONS[level]} {level} RISK</div>'
                f'<div class="prob">Cancellation probability: <b>{p:.1%}</b></div>'
                f'<div class="action">👉 {action}</div></div>', unsafe_allow_html=True)
