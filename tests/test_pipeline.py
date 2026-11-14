import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import pipeline  # noqa: E402


def _clean_sample():
    raw = pipeline.load_raw().sample(5000, random_state=0)
    return pipeline.add_features(pipeline.clean(raw))[0]


def test_no_missing_values_in_model_columns():
    df = _clean_sample()
    assert df[pipeline.NUM_FEATURES + pipeline.CAT_FEATURES].isna().sum().sum() == 0


def test_no_impossible_bookings():
    df = _clean_sample()
    assert (df["total_guests"] > 0).all()
    assert (df["total_nights"] > 0).all()
    assert df["adr"].between(0, 1000).all()


def test_no_leakage_columns_used():
    used = set(pipeline.NUM_FEATURES + pipeline.CAT_FEATURES)
    assert not used & {"reservation_status", "reservation_status_date", "assigned_room_type", "is_canceled"}
