"""Home page: what the project is, key results and how to use the website."""
import streamlit as st

from common import FIGURES, GITHUB, sidebar_footer

sidebar_footer()

st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown(
    "Hotels lose money when guests cancel, because the empty room often cannot be sold again. "
    "This website uses a machine learning model to predict, **at booking time**, how likely a booking "
    "is to be cancelled, so staff can act early: ask for a deposit, send a reminder or plan overbooking."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bookings analysed", "119,390", "86,625 after cleaning", delta_color="off")
c2.metric("Cancellation rate", "27.7%", "after cleaning", delta_color="off")
c3.metric("F1 score (test set)", "0.707", "target ≥ 0.70")
c4.metric("ROC-AUC (test set)", "0.896", "target ≥ 0.85")

st.subheader("What you can do here")
a, b = st.columns(2)
with a:
    st.page_link("views/predict.py", label="**Predict a booking** – enter one booking, get its risk and *why*", icon="🔮")
    st.page_link("views/batch.py", label="**Batch prediction** – upload many bookings, get a sorted risk list", icon="📋")
with b:
    st.page_link("views/insights.py", label="**Insights** – what drives cancellations in the data", icon="📊")
    st.page_link("views/performance.py", label="**Model performance** – compare models, try different thresholds", icon="⚙️")

st.subheader("How to read a prediction")
st.markdown(
    "| Risk level | Probability | Suggested action |\n"
    "|---|---|---|\n"
    "| 🟢 **LOW** | below 35% | No action needed |\n"
    "| 🟡 **MEDIUM** | 35% – 60% | Send a reminder and include the booking in overbooking estimates |\n"
    "| 🔴 **HIGH** | 60% or more | Contact the guest, ask for a deposit or offer a non-refundable rate |"
)

st.subheader("About the project")
left, right = st.columns([3, 2])
with left:
    st.markdown(
        "- **Data:** Hotel Booking Demand (Antonio, Almeida & Nunes, 2019) – a City Hotel in Lisbon and a "
        "Resort Hotel in the Algarve, July 2015 – August 2017. Licence CC BY 4.0.\n"
        "- **Model:** tuned Random Forest, chosen from 6 models (baseline, Logistic Regression, Naive Bayes, "
        "Decision Tree, Random Forest, Gradient Boosting).\n"
        "- **Only booking-time information is used** – nothing that is known only after the stay.\n"
        f"- **Code, notebooks and data:** [GitHub repository]({GITHUB})"
    )
    st.markdown(
        "**Team – Group 14:** Amna Nasser Alshamsi · Amna Ali Alhosani · Fatema Ahmed Alyileili  \n"
        "CDS 4003 – Data Science for Business Application"
    )
with right:
    img = FIGURES / "03_lead_time.png"
    if img.exists():
        st.image(str(img), caption="Bookings made far in advance cancel much more often.")
