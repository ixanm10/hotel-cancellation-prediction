"""Batch prediction: upload many bookings, get each one's risk, sorted from riskiest to safest."""
import pandas as pd
import streamlit as st

from common import ROOT, booking, load_bundle, sidebar_footer

sidebar_footer()
bundle = load_bundle()
model, top_countries = bundle["model"], bundle["top_countries"]
num, cat = bundle["num"], bundle["cat"]
SAMPLE = ROOT / "data" / "sample_bookings.csv"

st.title("📋 Batch prediction")
st.markdown(
    "Upload a CSV file of upcoming bookings (for example, tomorrow's arrivals). Every booking gets a "
    "cancellation probability, a risk level and a suggested action, sorted from **riskiest to safest**. "
    "The file uses the same column names as the original dataset – extra columns are kept and ignored."
)

with st.expander("Which columns does the file need?"):
    st.code(", ".join(booking.REQUIRED_COLUMNS), language=None)
    st.caption("`arrival_date_month` is the month name, e.g. July. Missing `children` counts as 0.")

left, right = st.columns([3, 2])
with left:
    uploaded = st.file_uploader("Upload bookings (CSV)", type="csv")
with right:
    st.write("No file? Try it with 40 example bookings:")
    use_sample = st.button("Use example bookings")
    if SAMPLE.exists():
        st.download_button("Download the example file", SAMPLE.read_bytes(), "sample_bookings.csv", "text/csv")

if uploaded is not None:
    st.session_state["batch_df"] = pd.read_csv(uploaded)
    st.session_state["batch_name"] = uploaded.name
elif use_sample:
    st.session_state["batch_df"] = pd.read_csv(SAMPLE)
    st.session_state["batch_name"] = "example bookings"

raw = st.session_state.get("batch_df")
if raw is None:
    st.stop()

missing = booking.missing_columns(raw)
if missing:
    st.error("The file is missing these columns: " + ", ".join(missing))
    st.stop()

X, problems = booking.features_from_raw(raw, top_countries)
result = raw.copy()
result["cancel_probability"] = pd.NA
if len(X):
    result.loc[X.index, "cancel_probability"] = model.predict_proba(X[num + cat])[:, 1]
result["cancel_probability"] = pd.to_numeric(result["cancel_probability"])
ok = problems == ""
result["risk_level"] = result["cancel_probability"].map(lambda p: booking.risk_level(p)[0] if pd.notna(p) else "not scored")
result["suggested_action"] = result["cancel_probability"].map(lambda p: booking.risk_level(p)[1] if pd.notna(p) else "")
nights = pd.to_numeric(result["stays_in_weekend_nights"], errors="coerce") + pd.to_numeric(result["stays_in_week_nights"], errors="coerce")
result["revenue_at_risk_eur"] = (result["cancel_probability"] * pd.to_numeric(result["adr"], errors="coerce") * nights).round(0)
result["problem"] = problems
result = result.sort_values("cancel_probability", ascending=False, na_position="last")

st.subheader(f"Results – {st.session_state['batch_name']}")
scored = result[ok.reindex(result.index)]
counts = scored["risk_level"].value_counts()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bookings scored", f"{len(scored):,}", f"{(~ok).sum()} could not be scored" if (~ok).any() else None, delta_color="off")
m2.metric("🔴 High risk", f"{counts.get('HIGH', 0):,}")
m3.metric("🟡 Medium risk", f"{counts.get('MEDIUM', 0):,}")
m4.metric("Expected cancellations", f"{scored['cancel_probability'].sum():,.0f}",
          f"€{scored['revenue_at_risk_eur'].sum():,.0f} revenue at risk", delta_color="off")

show_cols = [c for c in ["booking_id", "hotel", "arrival_date_month", "arrival_date_day_of_month", "lead_time",
                         "deposit_type", "market_segment", "adr"] if c in result.columns]
table = result[show_cols + ["cancel_probability", "risk_level", "revenue_at_risk_eur", "suggested_action", "problem"]].copy()
table["cancel_probability"] = (table["cancel_probability"] * 100).round(1)  # shown as a percentage
st.dataframe(
    table, hide_index=True,
    column_config={
        "cancel_probability": st.column_config.ProgressColumn("Cancel probability", format="%.0f%%", min_value=0, max_value=100),
        "revenue_at_risk_eur": st.column_config.NumberColumn("Revenue at risk (€)", format="€%d"),
    },
)
st.download_button("⬇️ Download results (CSV)", result.to_csv(index=False).encode(), "cancellation_risk_results.csv",
                   "text/csv", type="primary")
