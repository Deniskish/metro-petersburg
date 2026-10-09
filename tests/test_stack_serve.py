"""serve.forecast(model="stack"): стекинг в продакшн-прогнозе (этап 7, ч. 3)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config, contract
from src import eda_spb as E
from src import intrahour as I
from src import model as M
from src import serve as S
from src.stack import forecast as SF
from src.stack import meta as MT
from src.stack import __main__ as R
from test_serve import _future_garbage   # tests/ без __init__.py: модули импортируются по имени

NOW = "2026-07-08 13:59"      # t = 14:00: B1 lgbm_fold_2026-07, GRU stack_gru_2026-07
NOW30 = "2026-07-08 14:29"    # t = 14:30: незакрытый час 14 уже идёт
NEED = ["lgbm_fold_2026-05", "lgbm_fold_2026-06", "lgbm_fold_2026-07", "lgbm_holdout",
        "stack_gru_2026-07", "stack_gru_2026-09"]


def test_origin_half_hour_closes_at_29():
    assert SF.origin("2026-07-08 08:59") == pd.Timestamp("2026-07-08 09:00")
    assert SF.origin("2026-07-08 09:29") == pd.Timestamp("2026-07-08 09:30")
    assert SF.origin("2026-07-08 09:15") == pd.Timestamp("2026-07-08 09:00")
    for now in ("2026-07-08 08:59", "2026-07-08 09:29", "2026-07-08 09:44"):   # t0 стекинга = serve.origin
        t = SF.origin(now)
        assert t.floor("h") - pd.Timedelta(hours=1) == S.origin(now)


# --- Выбор GRU по дате -----------------------------------------------------------------------
GRU_END = {"stack_gru_2026-05": "2026-02-28", "stack_gru_2026-07": "2026-05-31", "stack_gru_2026-09": "2026-07-31",
           "stack_gru_final": "2026-09-30"}


def _fake_grus(path, names):
    for n in names:
        (path / n).mkdir()
        (path / n / "meta.json").write_text(json.dumps({"info": {"train_start": "2026-02-09", "train_end": GRU_END[n]}}))
    return path


@pytest.mark.parametrize("now, expected, oos", [
    ("2026-05-20 12:00", "stack_gru_2026-05", True),
    ("2026-07-08 13:59", "stack_gru_2026-07", True),
    ("2026-09-15 18:00", "stack_gru_2026-09", True),
    ("2026-02-20 12:00", "stack_gru_final", False),       # февраль: обученной до него GRU нет
])
def test_choose_gru(tmp_path, now, expected, oos):
    name, is_oos, warning = SF.choose_gru(now, _fake_grus(tmp_path, GRU_END))
    assert (name, is_oos) == (expected, oos)
    assert (warning is None) == oos


def test_choose_gru_never_after_now_minus_gap(tmp_path):
    d = _fake_grus(tmp_path, GRU_END)
    for now in pd.date_range("2026-02-10", "2026-10-10", freq="13h"):
        name, oos, warning = SF.choose_gru(now, d)
        if warning is None:
            assert oos and pd.Timestamp(GRU_END[name]) + pd.Timedelta(days=7) < now
        else:
            assert name == SF.GRU_FINAL and not oos


def test_choose_gru_without_models_is_unavailable(tmp_path):
    with pytest.raises(SF.Unavailable):
        SF.choose_gru(NOW, tmp_path)
    with pytest.raises(SF.Unavailable):              # есть только модели, обученные позже now, и нет финальной
        SF.choose_gru("2026-03-01 12:00", _fake_grus(tmp_path, ["stack_gru_2026-07"]))


# --- Реальные данные и модели -----------------------------------------------------------------
@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.SPB_15MIN.exists() and SF.PARAMS_JSON.exists()):
        pytest.skip("нет interim СПб, 15-минутных данных или reports/stack/stack_params.json")
    missing = [n for n in NEED if not (M.MODELS / n / "meta.json").exists()]
    if missing:
        pytest.skip(f"нет моделей {missing} — python -m src.model --save-folds, src.serve --train --holdout, src.stack --fit")
    return E.load()


@pytest.fixture(scope="module")
def hours(spb):
    return I.load_hours()


@pytest.fixture(scope="module")
def fc(spb, hours):
    return S.forecast(NOW, data=spb, hours=hours)


def _garbage_hours(h: pd.DataFrame, now: str) -> pd.DataFrame:
    """15-минутные четверти с ts ≥ t и часовой файл с часа t заменены шумом."""
    t = SF.origin(now).tz_localize(config.SPB_TZ).tz_convert("UTC")
    rng = np.random.default_rng(7)
    out = h.copy()
    for j, c in enumerate(I.QCOLS):
        after = (out.ts_utc + pd.Timedelta(minutes=15 * j)) >= t
        out.loc[after, c] = rng.integers(0, 5000, after.sum())
    out["y15"] = out[I.QCOLS].sum(1)
    later = out.ts_utc >= t.floor("h")
    out.loc[later, "y_hourly"] = rng.integers(0, 20000, later.sum())
    return out


@pytest.mark.parametrize("now", [NOW, NOW30])
def test_stack_does_not_see_after_now(spb, hours, now):
    """Шум в часовом потоке после t0 и в 15-минутных четвертях с ts ≥ t не меняет записи, причины и meta."""
    full = S.forecast(now, data=spb, hours=hours)
    alt = S.forecast(now, data=_future_garbage(spb, now), hours=_garbage_hours(hours, now))
    assert full.meta["model"] == "stack"
    assert full.records == alt.records
    assert full.explanations == alt.explanations
    assert full.meta == alt.meta


def test_stack_passes_contract(fc):
    preds = contract.validate_records(fc.records)
    assert len(preds) == 18 * 4
    assert {p.horizon_min for p in preds} == {30, 60, 90, 120}
    assert {r["ts"] for r in fc.records} == {f"2026-07-08T{x}:00+03:00" for x in ("14:00", "14:30", "15:00", "15:30")}
    assert all(r["q10"] <= r["q50"] <= r["q90"] for r in fc.records)
    assert all(r["model_version"] == "stack_30min_v1/lgbm_fold_2026-07+stack_gru_2026-07" for r in fc.records)
    assert len(fc.explanations) == 72 and all(len(e["reasons"]) == 3 for e in fc.explanations)
    keys = {(r["station_id"], r["ts"], r["horizon_min"]) for r in fc.records}
    assert keys == {(e["station_id"], e["ts"], e["horizon_min"]) for e in fc.explanations}
    assert len(S.table(fc)) == 72


def test_stack_meta_models_and_weights(fc):
    m = fc.meta
    assert m["model"] == "stack" and m["requested_model"] == "stack" and "fallback" not in m
    base = {b["name"]: b for b in m["base_models"]}
    assert set(base) == {"B1", "B2", "B3"}
    assert base["B1"]["bundle"] == "lgbm_fold_2026-07" and base["B1"]["out_of_sample"]
    assert base["B3"]["bundle"] == "stack_gru_2026-07" and base["B3"]["out_of_sample"]
    for q in ("q10", "q50", "q90"):
        for h in ("30", "60", "90", "120"):
            assert sum(b["weights"][q][h] for b in m["base_models"]) == pytest.approx(1.0)
    assert m["out_of_sample"] and not m["meta_model"]["out_of_sample"]     # июль: веса стекинга подбирались на нём
    assert SF.META_WARNING in m["warning"]
    assert m["meta_model"]["sha256"] == MT.load_params(SF.PARAMS_JSON)["sha256"]


def test_stack_september_fully_out_of_sample(spb, hours):
    fc = S.forecast("2026-09-15 18:00", data=spb, hours=hours)
    base = {b["name"]: b for b in fc.meta["base_models"]}
    assert base["B1"]["bundle"] == "lgbm_holdout" and base["B3"]["bundle"] == "stack_gru_2026-09"
    assert fc.meta["out_of_sample"] and fc.meta["meta_model"]["out_of_sample"] and fc.meta["warning"] is None
    contract.validate_records(fc.records)


@pytest.mark.parametrize("now, reason", [
    ("2026-06-10 12:59", "15-минутных данных"),       # июнь: 15-минутных данных нет
    ("2026-08-31 08:59", "15-минутных данных"),       # демо 31.08: август — только часовые данные
    ("2026-05-09 12:59", "необычные сутки"),          # 9 мая: 15-минутные данные есть, но праздник и событие
    ("2026-07-08 05:29", "вне часов 06–00"),          # t = 05:30
])
def test_fallback_to_lgbm(spb, hours, now, reason):
    fb = S.forecast(now, data=spb, hours=hours)
    lg = S.forecast(now, model="lgbm", data=spb)
    assert fb.meta["fallback"]["from"] == "stack" and reason in fb.meta["fallback"]["reason"]
    assert fb.meta["requested_model"] == "stack" and fb.meta["model"] == lg.meta["model"]
    assert fb.records == lg.records and fb.explanations == lg.explanations
    contract.validate_records(fb.records)


def test_fallback_without_stack_models(spb, hours, tmp_path):
    """Только бандлы LightGBM в models_dir: стекинг недоступен — та же LightGBM, без исключения."""
    for p in M.MODELS.glob("lgbm_*"):
        (tmp_path / p.name).symlink_to(p.resolve())
    fb = S.forecast(NOW, models_dir=tmp_path, data=spb, hours=hours)
    assert "GRU" in fb.meta["fallback"]["reason"]
    assert fb.records == S.forecast(NOW, model="lgbm", models_dir=tmp_path, data=spb).records


def test_stack_matches_research_pipeline(spb, hours):
    """Продакшн-путь = путь исследования: z базовых моделей совпадают с кэшем этапа 7, z стекинга — с meta.predict
    по итоговым параметрам (в кэше — перекрёстные веса май ↔ июль)."""
    path = R.CACHE / "cross.parquet"
    if not path.exists():
        pytest.skip("нет кэша прогнозов этапа 7 — python -m src.stack --fit")
    df, info = SF.frame(NOW, data=spb, hours=hours)
    cache = pd.read_parquet(path)
    cache = cache[cache.t == df.t.iloc[0]]
    m = df.merge(cache, on=["vestibule_id", "k"], suffixes=("", "_c"))
    assert len(m) >= 0.9 * len(cache) and len(m) > 50
    for name in ("b1", "b2", "b3"):
        for q in ("10", "50", "90"):
            np.testing.assert_allclose(m[f"{name}_{q}"], m[f"{name}_{q}_c"], rtol=0, atol=1e-6)
    base = cache.drop(columns=[c for c in cache.columns if c.startswith("stack")])
    ref = base[["vestibule_id", "k"]].join(MT.predict(base, info["params"]))
    mm = df.merge(ref, on=["vestibule_id", "k"], suffixes=("", "_r"))
    for q in ("10", "50", "90"):
        np.testing.assert_allclose(mm[f"stack_{q}"], mm[f"stack_{q}_r"], rtol=0, atol=1e-6)


def test_frozen_model_untouched():
    if not M.FINAL_JSON.exists():
        pytest.skip("нет model_final.json")
    M.check_frozen(M.load_config())
