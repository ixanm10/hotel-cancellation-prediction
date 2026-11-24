"""Insights: what drives cancellations, using the charts from the notebook."""
import streamlit as st

from common import FIGURES, banner, load_reference, sidebar_footer

sidebar_footer()
ref = load_reference()
df = ref["df"]

banner("insights", "📊 Insights from the data", "What makes a booking more likely to be cancelled.")
st.markdown(
    f"Based on **{len(df):,} clean bookings**, of which **{ref['cancel_rate']:.1%}** were cancelled. "
    "These are the patterns the model learned from."
)


def show(name, caption):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption)


st.subheader("1. The earlier the booking, the more likely it is cancelled")
a, b = st.columns([3, 2])
with a:
    show("03_lead_time.png", "Cancellation rate by lead time")
with b:
    by_band = (df.groupby("lead_time_band", observed=True)["is_canceled"].mean() * 100).round(1)
    st.dataframe(by_band.rename("cancelled (%)"))
    st.caption("Bookings made 0–7 days ahead rarely cancel; bookings made more than a year ahead cancel almost 5 times as often.")

st.subheader("2. Deposit type and special requests")
show("04_deposit_and_requests.png", "Cancellation rate by deposit type and number of special requests")
st.caption("'Non Refund' bookings cancel most – they are mostly group and travel-agent blocks. "
           "Guests who make special requests are committed and cancel much less.")

st.subheader("3. Lead time and deposit together")
show("15_heatmap_leadtime_deposit.png", "Cancellation rate (%) by lead time and deposit type")

st.subheader("4. Who cancels")
c, d = st.columns(2)
with c:
    show("05_market_segment.png", "Cancellation rate by market segment")
with d:
    show("06_month_and_hotel.png", "Cancellation rate by month and hotel")

st.subheader("5. What the model relies on most")
show("12_feature_importance.png", "Top 10 attributes – how much F1 drops when each one is shuffled")
