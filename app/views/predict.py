"""Predict one booking: risk, suggested action, why, and what-if simulator.

The form keeps its values in st.session_state, so:
  - the example buttons can fill the form in one click, and
  - after the first prediction the result updates live whenever an input changes.
"""
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from common import OLIVE, OXBLOOD, VELVET, banner, booking, load_bundle, load_reference, risk_card, sidebar_footer

sidebar_footer()
bundle = load_bundle()
ref = load_reference()
model, top_countries = bundle["model"], bundle["top_countries"]
num, cat = bundle["num"], bundle["cat"]
ss = st.session_state

DEFAULTS = dict(hotel="City Hotel", lead_time=60, weekend=1, week=2, adr=110.0, room="A", meal="BB",
                adults=2, children=0, babies=0, country="PRT", segment="Online TA", channel="TA/TO",
                deposit="No Deposit", customer="Transient", repeated=False, prev_c=0, prev_nc=0, changes=0,
                requests=0, parking=0, waiting=0)
EXAMPLES = {
    "🔴 Risky online booking": dict(hotel="City Hotel", lead_time=250, weekend=1, week=3, adr=120.0, segment="Online TA",
                                    channel="TA/TO", deposit="No Deposit", customer="Transient", requests=0, repeated=False),
    "🔴 Group block": dict(hotel="City Hotel", lead_time=200, weekend=0, week=3, adr=80.0, segment="Groups", channel="TA/TO",
                          deposit="Non Refund", customer="Transient-Party", requests=0, repeated=False),
    "🟡 Family at the resort": dict(hotel="Resort Hotel", lead_time=90, weekend=2, week=5, adr=180.0, adults=2, children=2,
                                   meal="HB", room="E", country="GBR", segment="Online TA", channel="TA/TO",
                                   deposit="No Deposit", requests=1, repeated=False),
    "🟢 Safe direct booking": dict(hotel="Resort Hotel", lead_time=5, weekend=1, week=2, adr=95.0, room="D", country="GBR",
                                  segment="Direct", channel="Direct", deposit="No Deposit", requests=2, parking=1,
                                  repeated=True, prev_nc=2),
}

for k, v in DEFAULTS.items():
    ss.setdefault(k, v)
ss.setdefault("arrival", date.today() + timedelta(days=60))
ss.setdefault("show_result", False)


def load_example(name):
    for k, v in {**DEFAULTS, **EXAMPLES[name]}.items():
        ss[k] = v
    ss["arrival"] = date.today() + timedelta(days=max(1, ss["lead_time"]))
    ss["show_result"] = True


banner("predict", "🔮 Predict a booking",
       f"Model: {bundle.get('model_name', 'best model')} · try an example or enter your own booking.")

st.markdown("**Try an example booking:**")
cols = st.columns(len(EXAMPLES))
for col, name in zip(cols, EXAMPLES):
    col.button(name, on_click=load_example, args=(name,))

c1, c2 = st.columns(2)
with c1:
    st.subheader("Booking")
    st.selectbox("Hotel", ["City Hotel", "Resort Hotel"], key="hotel")
    st.date_input("Arrival date", key="arrival")
    st.number_input("Lead time (days between booking and arrival)", 0, 800, key="lead_time")
    st.number_input("Weekend nights", 0, 20, key="weekend")
    st.number_input("Week nights", 0, 50, key="week")
    st.number_input("Average daily rate (€)", 0.0, 1000.0, step=5.0, key="adr")
    st.selectbox("Reserved room type", list("ABCDEFGHL"), key="room")
    st.selectbox("Meal", ["BB", "HB", "FB", "SC"], key="meal")
with c2:
    st.subheader("Guest and channel")
    st.number_input("Adults", 0, 10, key="adults")
    st.number_input("Children", 0, 10, key="children")
    st.number_input("Babies", 0, 10, key="babies")
    st.selectbox("Country", sorted(set(top_countries) | {"Other"}), key="country")
    st.selectbox("Market segment", ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"], key="segment")
    st.selectbox("Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS"], key="channel")
    st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"], key="deposit")
    st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"], key="customer")
    st.checkbox("Repeat guest", key="repeated")
    st.number_input("Previous cancellations", 0, 30, key="prev_c")
    st.number_input("Previous bookings not cancelled", 0, 80, key="prev_nc")
    st.number_input("Booking changes", 0, 20, key="changes")
    st.number_input("Special requests", 0, 5, key="requests")
    st.number_input("Car parking spaces", 0, 3, key="parking")
    st.number_input("Days on waiting list", 0, 400, key="waiting")

if st.button("Predict cancellation risk", type="primary"):
    ss["show_result"] = True
if not ss["show_result"]:
    st.stop()

nights = ss["weekend"] + ss["week"]
if nights == 0 or ss["adults"] + ss["children"] + ss["babies"] == 0:
    st.error("A booking needs at least 1 night and 1 guest.")
    st.stop()

raw = booking.raw_row(ss["hotel"], ss["arrival"], ss["lead_time"], ss["weekend"], ss["week"], ss["adults"],
                      ss["children"], ss["babies"], ss["meal"], ss["country"], ss["segment"], ss["channel"],
                      ss["repeated"], ss["prev_c"], ss["prev_nc"], ss["room"], ss["changes"], ss["deposit"],
                      ss["waiting"], ss["customer"], ss["adr"], ss["parking"], ss["requests"])
p, factors = booking.explain(model, raw, top_countries, ref["typical"], num, cat)
level, action = booking.risk_level(p)

st.divider()
st.caption("✨ The result updates live – change any input above and watch it move.")
risk_card(level, p, action)
m1, m2, m3 = st.columns(3)
m1.metric("Cancellation probability", f"{p:.1%}")
m2.metric("Revenue at risk", f"€{p * ss['adr'] * nights:,.0f}", help="probability × price per night × nights")
m3.metric("Stay value", f"€{ss['adr'] * nights:,.0f}", help="price per night × nights")

tab_why, tab_whatif, tab_lead = st.tabs(["❓ Why this result?", "🧪 What if…?", "📅 Booked earlier or later?"])

with tab_why:
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
        data = shown.assign(points=shown["effect"] * 100)
        chart = alt.Chart(data).mark_bar(cornerRadius=4).encode(
            x=alt.X("points:Q", title="Change in risk (percentage points)"),
            y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition("datum.points > 0", alt.value(VELVET), alt.value(OLIVE)),
            tooltip=["factor", "this booking", "typical booking", alt.Tooltip("points:Q", format="+.1f")])
        st.altair_chart(chart)
        st.caption("When many inputs point the same way, changing just one of them moves the risk only a little.")

with tab_whatif:
    base, actions = booking.what_if_actions(model, raw, top_countries, num, cat)
    st.markdown("How would the risk change if the guest did one of these things? "
                "Staff can use this to decide **what to encourage** (for example, offering a perk for booking directly).")
    if actions.empty:
        st.write("This booking already has all of these.")
    else:
        for _, r in actions.iterrows():
            new_level = booking.risk_level(r["new risk"])[0]
            arrow = "🔻" if r["change"] < 0 else "🔺"
            st.markdown(f"{arrow} **{r['action']}** → {r['new risk']:.0%} "
                        f"({'−' if r['change'] < 0 else '+'}{abs(r['change']) * 100:.1f} points, {new_level})")
        bars = pd.concat([pd.DataFrame([{"action": "Now", "new risk": base}]), actions[["action", "new risk"]]])
        chart = alt.Chart(bars).mark_bar(cornerRadius=4).encode(
            x=alt.X("new risk:Q", title="Cancellation risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("action:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.condition("datum.action == 'Now'", alt.value(OXBLOOD), alt.value(OLIVE)),
            tooltip=["action", alt.Tooltip("new risk:Q", format=".1%")])
        st.altair_chart(chart)
    st.caption("ℹ️ These are patterns learned from past bookings, not guaranteed effects.")

with tab_lead:
    curve = booking.lead_time_curve(model, raw, top_countries, num, cat)
    st.markdown("The same booking, if it had been made a different number of days before arrival. "
                "The big dot shows this booking.")
    line = alt.Chart(curve).mark_line(point=alt.OverlayMarkDef(color=OXBLOOD), color=OXBLOOD).encode(
        x=alt.X("lead_time:Q", title="Lead time (days before arrival)", scale=alt.Scale(domain=[0, 365])),
        y=alt.Y("risk:Q", title="Cancellation risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("lead_time:Q", title="lead time"), alt.Tooltip("risk:Q", format=".1%")])
    now = alt.Chart(pd.DataFrame({"lead_time": [ss["lead_time"]], "risk": [p]})).mark_point(
        size=220, filled=True, color=VELVET).encode(x="lead_time:Q", y="risk:Q")
    bands = alt.Chart(pd.DataFrame({"y": [booking.LOW_MAX, booking.MEDIUM_MAX]})).mark_rule(
        strokeDash=[4, 4], color="#B99A70").encode(y="y:Q")
    st.altair_chart(line + bands + now)
    st.caption("Dashed lines: 35% (Low → Medium) and 60% (Medium → High).")
