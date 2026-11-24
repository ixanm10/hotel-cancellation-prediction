"""
Hotel Booking Cancellation Prediction – project website (Streamlit).
Run locally:  streamlit run app/app.py

Pages (in app/views/):
  Home · Predict a booking · Batch prediction · Insights · Model performance
"""
import streamlit as st

st.set_page_config(page_title="Hotel Cancellation Risk", page_icon="🏨", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/predict.py", title="Predict a booking", icon="🔮"),
    st.Page("views/batch.py", title="Batch prediction", icon="📋"),
    st.Page("views/insights.py", title="Insights", icon="📊"),
    st.Page("views/performance.py", title="Model performance", icon="⚙️"),
]
st.navigation(pages).run()
