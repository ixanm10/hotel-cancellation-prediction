"""
Streamlit app - predict the cancellation risk of a new hotel booking.
Run locally:  streamlit run app/app.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
sys.path.append(str(ROOT / "src"))

st.set_page_config(page_title="Hotel Cancellation Risk", page_icon="🏨", layout="wide")


@st.cache_resource(show_spinner="Training the model for the first time (about 1 minute)...")
def load_bundle():
    # If the model file is not in the repo (e.g. first deploy), train it from the data.
    if not MODEL_PATH.exists():
        import pipeline
        pipeline.run(quick=True)
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model, top_countries = bundle["model"], bundle["top_countries"]

st.title("🏨 Hotel Booking Cancellation Risk")
st.caption(f"CDS 4003 – Group 3 | Model: {bundle.get('model_name', 'best model')}")

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

nights = weekend + week
if st.button("Predict cancellation risk", type="primary"):
    if nights == 0 or adults + children + babies == 0:
        st.error("A booking needs at least 1 night and 1 guest.")
        st.stop()
    prev_total = prev_c + prev_nc
    row = {
        "hotel": hotel, "lead_time": lead_time, "arrival_month_num": arrival.month,
        "arrival_date_week_number": arrival.isocalendar()[1], "arrival_date_day_of_month": arrival.day,
        "arrival_dow": arrival.weekday(), "stays_in_weekend_nights": weekend, "stays_in_week_nights": week,
        "total_nights": nights, "adults": adults, "children": children, "babies": babies,
        "total_guests": adults + children + babies, "has_kids": int(children + babies > 0),
        "is_repeated_guest": int(repeated), "previous_cancellations": prev_c,
        "previous_bookings_not_canceled": prev_nc, "prev_cancel_rate": prev_c / prev_total if prev_total else 0.0,
        "booking_changes": changes, "days_in_waiting_list": waiting, "adr": adr, "total_revenue": adr * nights,
        "weekend_ratio": weekend / nights, "required_car_parking_spaces": parking,
        "total_of_special_requests": requests, "is_summer": int(arrival.month in (6, 7, 8)),
        "meal": meal, "country_grp": country if country in top_countries else "Other", "market_segment": segment,
        "distribution_channel": channel, "reserved_room_type": room, "deposit_type": deposit, "customer_type": customer,
    }
    p = model.predict_proba(pd.DataFrame([row])[bundle["num"] + bundle["cat"]])[0, 1]
    level, action = (("HIGH", "Contact the guest, ask for a deposit or offer a non-refundable rate.") if p >= 0.6 else
                     ("MEDIUM", "Send a reminder and include the booking in overbooking estimates.") if p >= 0.35 else
                     ("LOW", "No action needed."))
    m1, m2, m3 = st.columns(3)
    m1.metric("Cancellation probability", f"{p:.1%}")
    m2.metric("Risk level", level)
    m3.metric("Revenue at risk", f"€{p * adr * nights:,.0f}")
    st.info(f"Suggested action: {action}")
