"""Model performance: model comparison and an interactive threshold / cost explorer."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import FIGURES, banner, booking, load_bundle, load_reference, sidebar_footer

sidebar_footer()
bundle = load_bundle()
ref = load_reference()
y, proba = ref["y_test"], ref["proba_test"]

banner("performance", "⚙️ Model performance", "How good is the model – and what threshold should the hotel use?")

st.subheader("1. Six models compared (hold-out test set, 17,325 bookings)")
comparison = pd.DataFrame(
    [["Random Forest (tuned) – final", 0.820, 0.643, 0.786, 0.707, 0.896],
     ["Gradient Boosted Trees (tuned)", 0.811, 0.619, 0.820, 0.706, 0.899],
     ["Decision Tree (tuned)", 0.762, 0.546, 0.824, 0.657, 0.865],
     ["Logistic Regression", 0.743, 0.525, 0.768, 0.623, 0.832],
     ["Naive Bayes", 0.555, 0.371, 0.874, 0.521, 0.761],
     ["Baseline (always 'not cancelled')", 0.723, 0.000, 0.000, 0.000, 0.500]],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
st.dataframe(comparison, hide_index=True)
st.caption("The baseline is 72% accurate but catches **no** cancellations – that is why we judge models by F1 and recall, not accuracy.")

a, b = st.columns(2)
with a:
    path = FIGURES / "10_roc_curve.png"
    if path.exists():
        st.image(str(path), caption="ROC curves – the further from the diagonal, the better the model ranks risky bookings")
with b:
    path = FIGURES / "11_lift_chart.png"
    if path.exists():
        st.image(str(path), caption="Lift chart – the riskiest 20% of bookings contain about 56% of all cancellations")

st.subheader("2. Choose your own threshold")
st.markdown(
    "The model gives every booking a probability. The **threshold** decides when we call it a *likely cancellation*. "
    "A lower threshold catches more cancellations but raises more false alarms. Move the slider and change the costs "
    f"to see the effect on the test set ({len(y):,} bookings, model: {bundle.get('model_name', 'final model')})."
)
c1, c2, c3 = st.columns(3)
threshold = c1.slider("Threshold", 0.10, 0.90, 0.40, 0.05)
cost_fn = c2.number_input("Cost of a missed cancellation (€)", 0, 2000, 180, 10,
                          help="Lost revenue when a cancelled room is not resold (report assumption: €180)")
cost_fp = c3.number_input("Cost of a false alarm (€)", 0, 500, 20, 5,
                          help="Staff time and a possibly annoyed guest (report assumption: €20)")

row = booking.threshold_table(y, proba, [threshold], cost_fn, cost_fp).iloc[0]
baseline_cost = int((y == 1).sum()) * cost_fn
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recall", f"{row['recall']:.1%}", help="Share of real cancellations the model catches")
m2.metric("Precision", f"{row['precision']:.1%}", help="Share of 'will cancel' predictions that were right")
m3.metric("Total cost of errors", f"€{row['cost (€)']:,.0f}")
m4.metric("Saving vs doing nothing", f"€{baseline_cost - row['cost (€)']:,.0f}")

cm = pd.DataFrame([[row["correct stays"], row["false alarms"]], [row["missed"], row["caught"]]],
                  index=["Actually stayed", "Actually cancelled"],
                  columns=["Predicted: stays", "Predicted: cancels"])
left, right = st.columns([2, 3])
with left:
    st.markdown("**Confusion matrix at this threshold**")
    st.dataframe(cm)
with right:
    grid = booking.threshold_table(y, proba, np.round(np.arange(0.10, 0.91, 0.05), 2), cost_fn, cost_fp)
    best_t = grid.loc[grid["cost (€)"].idxmin(), "threshold"]
    st.markdown(f"**Total cost of errors by threshold** – cheapest at **{best_t:.2f}** with these costs")
    costs = grid.rename(columns={"cost (€)": "cost"})
    x = alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.1, 0.9]))
    line = alt.Chart(costs).mark_line(point=True, color="#5A302C").encode(
        x=x, y=alt.Y("cost:Q", title="Total cost of errors (€)", axis=alt.Axis(format=",.0f")),
        tooltip=[alt.Tooltip("threshold:Q"), alt.Tooltip("cost:Q", format=",.0f", title="cost (€)")])
    marker = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="#7A1020", strokeDash=[4, 4]).encode(x=x)
    st.altair_chart(line + marker)

st.markdown("**Recall and precision by threshold** (dashed line = your threshold)")
long = grid.melt(id_vars="threshold", value_vars=["recall", "precision"], var_name="measure", value_name="value")
rp = alt.Chart(long).mark_line(point=True).encode(
    x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.1, 0.9])),
    y=alt.Y("value:Q", title="", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    color=alt.Color("measure:N", title=None, scale=alt.Scale(domain=["recall", "precision"], range=["#5A302C", "#B99A70"])),
    tooltip=[alt.Tooltip("threshold:Q"), alt.Tooltip("measure:N"), alt.Tooltip("value:Q", format=".1%")])
marker2 = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="#7A1020", strokeDash=[4, 4]).encode(
    x=alt.X("threshold:Q", scale=alt.Scale(domain=[0.1, 0.9])))
st.altair_chart(rp + marker2)
st.caption("Our report recommends a threshold of **0.40**: it catches about 87% of cancellations while keeping false alarms manageable.")
