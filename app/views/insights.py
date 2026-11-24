"""Insights: what drives cancellations, using the charts from the notebook."""
import altair as alt
import pandas as pd
import streamlit as st

from common import FIGURES, OLIVE, OXBLOOD, PAPER, VELVET, banner, load_reference, sidebar_footer

sidebar_footer()
ref = load_reference()
df = ref["df"]

banner("insights", "📊 Insights from the data", "What makes a booking more likely to be cancelled.")
st.markdown(
    f"Based on **{len(df):,} clean bookings**, of which **{ref['cancel_rate']:.1%}** were cancelled. "
    "These are the patterns the model learned from."
)


# ---------------------------------------------------------------- interactive explorer
st.subheader("🔎 Explore the data yourself")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
f1, f2, f3, f4 = st.columns(4)
hotels = f1.multiselect("Hotel", sorted(df["hotel"].unique()), default=sorted(df["hotel"].unique()))
segments = f2.multiselect("Market segment", sorted(df["market_segment"].unique()), placeholder="All segments")
deposits = f3.multiselect("Deposit type", sorted(df["deposit_type"].unique()), placeholder="All deposit types")
months = f4.select_slider("Arrival months", options=MONTHS, value=("Jan", "Dec"))

sel = df[df["hotel"].isin(hotels or df["hotel"].unique())]
if segments:
    sel = sel[sel["market_segment"].isin(segments)]
if deposits:
    sel = sel[sel["deposit_type"].isin(deposits)]
m_from, m_to = MONTHS.index(months[0]) + 1, MONTHS.index(months[1]) + 1
sel = sel[sel["arrival_month_num"].between(m_from, m_to)]

if sel.empty:
    st.warning("No bookings match these filters.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bookings", f"{len(sel):,}")
    k2.metric("Cancellation rate", f"{sel['is_canceled'].mean():.1%}")
    if len(sel) < len(df):
        k2.caption(f"{(sel['is_canceled'].mean() - ref['cancel_rate']) * 100:+.1f} points vs all bookings ({ref['cancel_rate']:.1%})")
    k3.metric("Average price per night", f"€{sel['adr'].mean():,.0f}")
    k4.metric("Revenue lost to cancellations", f"€{(sel['total_revenue'] * sel['is_canceled']).sum() / 1e6:,.2f}M")

    GROUPS = {"Lead time": "lead_time_band", "Market segment": "market_segment", "Deposit type": "deposit_type",
              "Customer type": "customer_type", "Special requests": "total_of_special_requests",
              "Arrival month": "arrival_month_num", "Hotel": "hotel", "Repeat guest": "is_repeated_guest"}
    by = st.radio("Compare cancellation rate by", list(GROUPS), horizontal=True)
    col = GROUPS[by]
    g = (sel.groupby(col, observed=True)["is_canceled"].agg(rate="mean", bookings="size").reset_index())
    g[col] = g[col].astype(str)
    if col == "arrival_month_num":
        g[col] = g[col].map(lambda m: MONTHS[int(m) - 1])
    order = list(g[col]) if col in ("lead_time_band", "arrival_month_num", "total_of_special_requests") else "-y"
    avg = sel["is_canceled"].mean()
    bars = alt.Chart(g).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X(f"{col}:N", title=by, sort=order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Cancellation rate", axis=alt.Axis(format="%")),
        color=alt.condition(f"datum.rate > {avg}", alt.value(VELVET), alt.value(OLIVE)),
        tooltip=[alt.Tooltip(f"{col}:N", title=by), alt.Tooltip("rate:Q", format=".1%", title="cancelled"),
                 alt.Tooltip("bookings:Q", format=",")])
    rule = alt.Chart(pd.DataFrame({"avg": [avg]})).mark_rule(strokeDash=[5, 5], color=PAPER, size=2).encode(y="avg:Q")
    st.altair_chart(bars + rule)
    st.caption("Red bars are above the average for your selection (dashed line), olive bars below. Hover for details.")

    trend = (sel.groupby(["arrival_month_num", "hotel"])["is_canceled"].mean().reset_index()
             .assign(month=lambda d: d["arrival_month_num"].map(lambda m: MONTHS[int(m) - 1])))
    line = alt.Chart(trend).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("month:N", sort=MONTHS, title="Arrival month"),
        y=alt.Y("is_canceled:Q", title="Cancellation rate", axis=alt.Axis(format="%")),
        color=alt.Color("hotel:N", scale=alt.Scale(range=[OXBLOOD, OLIVE]), title=None),
        tooltip=["hotel", "month", alt.Tooltip("is_canceled:Q", format=".1%", title="cancelled")])
    st.markdown("**Cancellation rate by month**")
    st.altair_chart(line)

st.divider()
st.subheader("📈 Key charts from our analysis")


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
