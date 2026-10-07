import pytest

from src import config
from src.contract import ContractError, make_stub, validate_records

GOOD = {
    "station_id": "vosstaniya",
    "ts": "2026-11-14T18:00:00+03:00",
    "horizon_min": 30,
    "q10": 2100, "q50": 2600, "q90": 3150,
    "baseline": 1850,
    "is_anomaly": True,
    "model_version": "lgbm_events_v1",
}


def test_example_from_tz_is_valid():
    (p,) = validate_records([GOOD])
    assert p.ts.utcoffset().total_seconds() == 3 * 3600


def test_stub_is_valid():
    stub = make_stub()
    preds = validate_records(stub)
    assert len(preds) == len(config.SPB_LINE1_STATIONS) * 72 * 4
    assert {p.station_id for p in preds} == set(config.SPB_LINE1_STATIONS)
    assert {p.ts.date().isoformat() for p in preds} == {"2026-11-12"}
    assert preds[0].ts.weekday() == 3  # четверг
    assert any(p.is_anomaly for p in preds) and not all(p.is_anomaly for p in preds)


def test_stub_is_deterministic():
    assert make_stub() == make_stub()


@pytest.mark.parametrize("patch", [
    {"q10": 2700},                         # q10 > q50
    {"ts": "2026-11-14T18:00:00"},         # нет смещения
    {"ts": "2026-11-14T18:05:00+03:00"},   # не граница слота
    {"horizon_min": 0},
    {"horizon_min": 135},
    {"horizon_min": 40},                   # не кратно 15
    {"q50": -1},
    {"is_anomaly": "true"},
    {"station_id": ""},
    {"extra_field": 1},
])
def test_bad_record_rejected(patch):
    with pytest.raises(ContractError):
        validate_records([{**GOOD, **patch}])


def test_missing_field_rejected():
    rec = {k: v for k, v in GOOD.items() if k != "baseline"}
    with pytest.raises(ContractError):
        validate_records([rec])


def test_duplicate_key_rejected():
    with pytest.raises(ContractError, match="дубликат"):
        validate_records([GOOD, GOOD])


def test_all_errors_collected():
    with pytest.raises(ContractError) as exc:
        validate_records([{**GOOD, "q10": 9999}, GOOD, {**GOOD, "horizon_min": 7}])
    assert len(exc.value.errors) == 2
