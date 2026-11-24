"""
Booking helpers used by the Streamlit website (app/).

Everything here is plain pandas, so it can be tested without Streamlit:
  - features_from_raw()  turns bookings in the original dataset format into model features
  - risk_level()         turns a probability into Low / Medium / High and a suggested action
  - typical_booking()    the most common value of each input (used to explain a prediction)
  - explain()            which inputs push one booking's risk up or down
  - threshold_table()    recall, precision and cost of errors at different thresholds
"""
import numpy as np
import pandas as pd

import pipeline

MONTHS = list(pipeline.MONTH_MAP)  # January ... December

# Columns a batch file must have (same names as the original Kaggle dataset).
REQUIRED_COLUMNS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "booking_changes", "deposit_type",
    "days_in_waiting_list", "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
]

# Risk bands shown to staff.
LOW_MAX, MEDIUM_MAX = 0.35, 0.60


def risk_level(p):
    """Probability -> (level, suggested action)."""
    if p >= MEDIUM_MAX:
        return "HIGH", "Contact the guest, ask for a deposit or offer a non-refundable rate."
    if p >= LOW_MAX:
        return "MEDIUM", "Send a reminder and include the booking in overbooking estimates."
    return "LOW", "No action needed."


def missing_columns(raw):
    return [c for c in REQUIRED_COLUMNS if c not in raw.columns]


def features_from_raw(raw, top_countries):
    """Bookings in the original dataset format -> (features for the model, list of problems per row).

    Applies the same cleaning and feature steps as the pipeline, but never drops rows:
    rows that cannot be scored get a message in `problems` instead.
    """
    df = raw[REQUIRED_COLUMNS].copy()
    given_week = pd.to_numeric(raw["arrival_date_week_number"], errors="coerce") \
        if "arrival_date_week_number" in raw.columns else None
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["meal"] = df["meal"].replace("Undefined", "SC")
    num_cols = [c for c in REQUIRED_COLUMNS if c not in
                ("hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
                 "reserved_room_type", "deposit_type", "customer_type")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["arrival_month_num"] = df["arrival_date_month"].map(pipeline.MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month_num"],
                                             day=df["arrival_date_day_of_month"]), errors="coerce")
    # Use the dataset's own week number when the file has it, otherwise the calendar week.
    df["arrival_date_week_number"] = df["arrival_date"].dt.isocalendar().week.astype("float")
    if given_week is not None:
        df["arrival_date_week_number"] = given_week.fillna(df["arrival_date_week_number"])

    problems = pd.Series("", index=df.index)
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    guests = df["adults"] + df["children"] + df["babies"]
    problems[df[num_cols].isna().any(axis=1)] = "missing or non-numeric value"
    problems[df["arrival_date"].isna()] = "invalid arrival date"
    problems[nights <= 0] = "booking has 0 nights"
    problems[guests <= 0] = "booking has 0 guests"
    problems[(df["adr"] < 0) | (df["adr"] > 1000)] = "price (adr) outside 0–1000"

    ok = problems == ""
    feats, _ = pipeline.add_features(df[ok].assign(assigned_room_type=df.loc[ok, "reserved_room_type"]),
                                     top_countries=top_countries)
    X = feats[pipeline.NUM_FEATURES + pipeline.CAT_FEATURES]
    return X, problems


def raw_row(hotel, arrival, lead_time, weekend, week, adults, children, babies, meal, country, segment,
            channel, repeated, prev_c, prev_nc, room, changes, deposit, waiting, customer, adr, parking, requests):
    """One booking from the website form -> one-row DataFrame in the original dataset format."""
    return pd.DataFrame([{
        "hotel": hotel, "lead_time": lead_time, "arrival_date_year": arrival.year,
        "arrival_date_month": MONTHS[arrival.month - 1], "arrival_date_day_of_month": arrival.day,
        "stays_in_weekend_nights": weekend, "stays_in_week_nights": week, "adults": adults,
        "children": children, "babies": babies, "meal": meal, "country": country,
        "market_segment": segment, "distribution_channel": channel, "is_repeated_guest": int(repeated),
        "previous_cancellations": prev_c, "previous_bookings_not_canceled": prev_nc,
        "reserved_room_type": room, "booking_changes": changes, "deposit_type": deposit,
        "days_in_waiting_list": waiting, "customer_type": customer, "adr": adr,
        "required_car_parking_spaces": parking, "total_of_special_requests": requests,
    }])


def typical_booking(clean_df):
    """Most common value (categories) or median (numbers) of each input in the training data."""
    typical = {}
    for c in REQUIRED_COLUMNS:
        if c in ("arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"):
            continue
        s = clean_df[c]
        typical[c] = float(s.median()) if pd.api.types.is_numeric_dtype(s) else s.mode()[0]
    return typical


# Inputs explained one at a time: label -> columns changed together.
FACTORS = {
    "Lead time": ["lead_time"],
    "Deposit type": ["deposit_type"],
    "Market segment": ["market_segment"],
    "Distribution channel": ["distribution_channel"],
    "Customer type": ["customer_type"],
    "Special requests": ["total_of_special_requests"],
    "Car parking": ["required_car_parking_spaces"],
    "Previous cancellations": ["previous_cancellations"],
    "Repeat guest": ["is_repeated_guest"],
    "Booking changes": ["booking_changes"],
    "Price per night": ["adr"],
    "Country": ["country"],
    "Hotel": ["hotel"],
    "Length of stay": ["stays_in_weekend_nights", "stays_in_week_nights"],
}


def _same(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def explain(model, raw, top_countries, typical, num, cat):
    """For one booking: how much each input changes the risk compared with a typical value.

    Method (simple "what-if"): replace one input with its typical value, predict again,
    and compare. A positive change means this input makes the booking MORE likely to cancel.
    """
    def proba(r):
        X, problems = features_from_raw(r, top_countries)
        return float(model.predict_proba(X[num + cat])[0, 1]) if (problems == "").all() else np.nan

    base = proba(raw)
    rows = []
    for label, cols in FACTORS.items():
        changed = raw.copy()
        for c in cols:
            changed[c] = typical[c]
        if all(_same(raw.iloc[0][c], changed.iloc[0][c]) for c in cols):
            continue  # already typical
        p = proba(changed)
        if np.isnan(p):
            continue
        value = " + ".join(str(raw.iloc[0][c]) for c in cols)
        typ = " + ".join(str(int(typical[c]) if isinstance(typical[c], float) and typical[c].is_integer()
                             else typical[c]) for c in cols)
        rows.append({"factor": label, "this booking": value, "typical booking": typ, "effect": base - p})
    out = pd.DataFrame(rows, columns=["factor", "this booking", "typical booking", "effect"])
    return base, out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def threshold_table(y_true, proba, thresholds, cost_fn=pipeline.COST_FN, cost_fp=pipeline.COST_FP):
    """Recall, precision, false alarms, missed cancellations and cost at each threshold."""
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = proba >= t
        tp = int((pred & (y_true == 1)).sum()); fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum()); tn = int((~pred & (y_true == 0)).sum())
        rows.append({"threshold": round(float(t), 2), "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "precision": tp / (tp + fp) if tp + fp else 0.0, "accuracy": (tp + tn) / len(y_true),
                     "caught": tp, "missed": fn, "false alarms": fp, "correct stays": tn,
                     "cost (€)": fn * cost_fn + fp * cost_fp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- what-if simulator
# Actions staff could encourage, and how each one changes the booking (columns -> new value).
# Parking is left out on purpose: almost no guest with a parking space cancelled in the data, so the model
# treats parking as a near-guarantee - true for past bookings, but not a realistic action to recommend.
WHAT_IF_ACTIONS = {
    "Guest books directly with the hotel": {"market_segment": "Direct", "distribution_channel": "Direct"},
    "Guest adds a special request": {"total_of_special_requests": lambda r: r["total_of_special_requests"] + 1},
    "Guest books 30 days before arrival": {"lead_time": lambda r: min(30, r["lead_time"])},
    "Guest is a returning customer": {"is_repeated_guest": 1},
}


def _predict(model, raw, top_countries, num, cat):
    X, problems = features_from_raw(raw, top_countries)
    p = pd.Series(np.nan, index=raw.index)
    if len(X):
        p[X.index] = model.predict_proba(X[num + cat])[:, 1]
    return p


def what_if_actions(model, raw, top_countries, num, cat):
    """For one booking: the new risk after each action in WHAT_IF_ACTIONS (only actions that change something)."""
    base = float(_predict(model, raw, top_countries, num, cat).iloc[0])
    row = raw.iloc[0]
    rows = []
    for name, changes in WHAT_IF_ACTIONS.items():
        new = raw.copy()
        for col, val in changes.items():
            new[col] = val(row) if callable(val) else val
        if all(_same(new.iloc[0][c], row[c]) for c in changes):
            continue  # this booking already has it
        p = float(_predict(model, new, top_countries, num, cat).iloc[0])
        rows.append({"action": name, "new risk": p, "change": p - base})
    out = pd.DataFrame(rows, columns=["action", "new risk", "change"])
    return base, out.sort_values("change").reset_index(drop=True)


def lead_time_curve(model, raw, top_countries, num, cat, lead_times=range(0, 366, 15)):
    """Risk of the same booking if it had been made a different number of days before arrival."""
    many = pd.concat([raw.assign(lead_time=lt) for lt in lead_times], ignore_index=True)
    return pd.DataFrame({"lead_time": list(lead_times),
                         "risk": _predict(model, many, top_countries, num, cat).to_numpy()})
