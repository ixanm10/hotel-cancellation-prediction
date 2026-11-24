import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import booking   # noqa: E402
import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_risk_levels():
    assert booking.risk_level(0.10)[0] == "LOW"
    assert booking.risk_level(0.40)[0] == "MEDIUM"
    assert booking.risk_level(0.75)[0] == "HIGH"


def test_example_file_has_all_columns_and_scores():
    raw = pd.read_csv(ROOT / "data" / "sample_bookings.csv")
    assert booking.missing_columns(raw) == []
    X, problems = booking.features_from_raw(raw, top_countries=["PRT", "GBR", "FRA"])
    assert (problems == "").all()
    assert list(X.columns) == pipeline.NUM_FEATURES + pipeline.CAT_FEATURES
    assert X.isna().sum().sum() == 0


def test_invalid_bookings_are_flagged_not_dropped():
    row = booking.raw_row("City Hotel", date(2027, 1, 23), 60, 0, 0, 2, 0, 0, "BB", "PRT", "Online TA", "TA/TO",
                          False, 0, 0, "A", 0, "No Deposit", 0, "Transient", 110.0, 0, 0)
    X, problems = booking.features_from_raw(row, top_countries=["PRT"])
    assert len(X) == 0 and problems.iloc[0] == "booking has 0 nights"


def test_threshold_table_counts_add_up():
    y = pd.Series([1, 0, 1, 0]).to_numpy()
    t = booking.threshold_table(y, pd.Series([0.9, 0.8, 0.2, 0.1]).to_numpy(), [0.5]).iloc[0]
    assert (t["caught"], t["false alarms"], t["missed"], t["correct stays"]) == (1, 1, 1, 1)
    assert t["cost (€)"] == pipeline.COST_FN + pipeline.COST_FP
