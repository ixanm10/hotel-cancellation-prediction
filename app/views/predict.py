"""Predict one booking: risk, suggested action and which inputs make it risky."""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from common import banner, booking, load_bundle, load_reference, risk_card, sidebar_footer

sidebar_footer()
bundle = load_bundle()
ref = load_reference()
model, top_countries = bundle["model"], bundle["top_countries"]
num, cat = bundle["num"], bundle["cat"]

banner("predict", "🔮 Predict a booking", f"Model: {bundle.get('model_name', 'best model')} · enter the booking details and click Predict.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Booking")
    hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    arrival = st.date_input("Arrival date", date.today() + timedelta(days=60))
    lead_time = st.number_input("Lead time (days between booking and arrival)", 0, 800, 60)
    weekend = st.number_input("Weekend nights", 0, 20, 1)
    week = st.number_input("Week nights", 0, 50, 2)
    adr = st.number_input("Average daily rate (€)", 0.0, 1000.0, 110.0, step=5.0)
    room = st.selectbox("Reserved room type", list("ABCDEFGHL"))
    meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC"])
with c2:
    st.subheader("Guest and channel")
    adults = st.number_input("Adults", 0, 10, 2)
    children = st.number_input("Children", 0, 10, 0)
    babies = st.number_input("Babies", 0, 10, 0)
    country = st.selectbox("Country", sorted(set(top_countries) | {"Other"}))
    segment = st.selectbox("Market segment", ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"])
    channel = st.selectbox("Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS"])
    deposit = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
    customer = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
    repeated = st.checkbox("Repeat guest")
    prev_c = st.number_input("Previous cancellations", 0, 30, 0)
    prev_nc = st.number_input("Previous bookings not cancelled", 0, 80, 0)
    changes = st.number_input("Booking changes", 0, 20, 0)
    requests = st.number_input("Special requests", 0, 5, 0)
    parking = st.number_input("Car parking spaces", 0, 3, 0)
    waiting = st.number_input("Days on waiting list", 0, 400, 0)

if st.button("Predict cancellation risk", type="primary"):
    nights = weekend + week
    if nights == 0 or adults + children + babies == 0:
        st.error("A booking needs at least 1 night and 1 guest.")
        st.stop()

    raw = booking.raw_row(hotel, arrival, lead_time, weekend, week, adults, children, babies, meal, country,
                          segment, channel, repeated, prev_c, prev_nc, room, changes, deposit, waiting,
                          customer, adr, parking, requests)
    p, factors = booking.explain(model, raw, top_countries, ref["typical"], num, cat)
    level, action = booking.risk_level(p)

    risk_card(level, p, action)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cancellation probability", f"{p:.1%}")
    m2.metric("Revenue at risk", f"€{p * adr * nights:,.0f}", help="probability × price per night × nights")
    m3.metric("Stay value", f"€{adr * nights:,.0f}", help="price per night × nights")

    st.subheader("Why this result?")
    st.caption(
        "We change **one input at a time** to its most common value in our data (for example lead time 50 days, "
        "no deposit, Online TA) and let the model predict again. The difference shows how much that input matters "
        "for this booking. 🔺 = this input makes cancellation **more** likely, 🔻 = **less** likely."
    )
    shown = factors[factors["effect"].abs() >= 0.005].head(6)
    if shown.empty:
        st.write("This booking is very close to a typical booking, so no single input stands out.")
    else:
        for _, r in shown.iterrows():
            arrow = "🔺" if r["effect"] > 0 else "🔻"
            st.markdown(f"{arrow} **{r['factor']}**: {r['this booking']} (typical: {r['typical booking']}) → "
                        f"{'+' if r['effect'] > 0 else '−'}{abs(r['effect']) * 100:.1f} percentage points")
        chart = shown.set_index("factor")["effect"].mul(100).rename("change in risk (points)")
        st.bar_chart(chart, horizontal=True, color="#74613F")
        st.caption("When many inputs point the same way, changing just one of them moves the risk only a little.")
