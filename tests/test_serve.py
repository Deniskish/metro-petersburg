import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import config, contract
from src import eda_spb as E
from src import model as M
from src import serve as S


def test_origin_hour_closes_at_59():
    assert S.origin("2026-08-31 08:59") == pd.Timestamp("2026-08-31 08:00")
    assert S.origin("2026-08-31 08:30") == pd.Timestamp("2026-08-31 07:00")
    assert S.origin("2026-08-31 09:00") == pd.Timestamp("2026-08-31 08:00")
    assert S.origin(pd.Timestamp("2026-08-31 05:59", tz="UTC")) == pd.Timestamp("2026-08-31 08:00")   # 08:59 MSK


# --- Выбор модели -----------------------------------------------------------------
TRAIN_END = {"lgbm_fold_2026-05": "2026-04-23", "lgbm_fold_2026-06": "2026-05-24", "lgbm_fold_2026-07": "2026-06-23",
             "lgbm_fold_2026-08": "2026-07-24", "lgbm_holdout": "2026-08-24", "lgbm_final": "2026-09-29"}


@pytest.fixture()
def fake_models(tmp_path):
    for name, end in TRAIN_END.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text(json.dumps({"name": name, "train_end": end}))
    return tmp_path


@pytest.mark.parametrize("now, expected", [
    ("2026-06-27 20:59", "lgbm_fold_2026-06"),
    ("2026-08-31 08:59", "lgbm_holdout"),
    ("2026-08-30 12:00", "lgbm_fold_2026-08"),      # holdout обучен по 24.08: 7 суток ещё не прошли
    ("2026-05-01 00:00", "lgbm_fold_2026-05"),
    ("2026-10-07 09:00", "lgbm_final"),
])
def test_choose_model_auto(fake_models, now, expected):
    name, oos, warning = S.choose_model(now, fake_models)
    assert name == expected and oos and warning is None


def test_auto_never_takes_model_trained_after_now_minus_gap(fake_models):
    """Перебор моментов: auto не берёт модель с train_end ≥ now − 7 суток; иначе — lgbm_final с предупреждением."""
    for now in pd.date_range("2026-04-20", "2026-10-10", freq="7h"):
        name, oos, warning = S.choose_model(now, fake_models)
        end = pd.Timestamp(TRAIN_END[name])
        if warning is None:
            assert oos and end + pd.Timedelta(days=7) < now, (now, name)
        else:
            assert name == S.FULL and not oos
            assert all(pd.Timestamp(e) + pd.Timedelta(days=7) >= now for e in TRAIN_END.values()), now
    assert S.choose_model("2026-04-30 00:00", fake_models) == (S.FULL, False, S.WARNING)   # ровно 7 суток — рано


def test_explicit_model_warns_if_not_out_of_sample(fake_models):
    assert S.choose_model("2026-08-31 08:59", fake_models, "lgbm_final") == ("lgbm_final", False, S.WARNING)
    assert S.choose_model("2026-08-31 08:59", fake_models, "lgbm_holdout") == ("lgbm_holdout", True, None)


# --- Прогноз на реальных данных ------------------------------------------------------
@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists() and M.FINAL_JSON.exists()):
        pytest.skip("нет interim СПб или model_final.json")
    return E.load()


@pytest.fixture(scope="module")
def tiny(spb, tmp_path_factory):
    """Финальная конфигурация с 20 деревьями — модель-малютка для проверки serve."""
    cfg = {**M.load_config(), "max_rounds": 20, "es_rounds": 5}
    d = tmp_path_factory.mktemp("models")
    S.train_full(cfg, d)
    return d


def _future_garbage(d: E.SpbData, now: str) -> E.SpbData:
    """Поток после конца часа t0 заменён шумом, инцидент после t0 снят — на прогноз это влиять не должно."""
    t0 = S.origin(now).tz_localize(config.SPB_TZ).tz_convert("UTC")
    g = d.grid
    after = (g.ts > t0)[:, None]
    noise = np.random.default_rng(3).integers(0, 20_000, g.entries.shape).astype(float)
    regular = d.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    grid = E.make_grid(g.ts, np.where(after, noise, g.entries), g.closed, np.where(after, g.closed, g.flagged), regular)
    df = d.df.copy()
    df.loc[df.ts_utc > t0, ["entries", "y"]] = 1e6
    return replace(d, grid=grid, df=df)


@pytest.mark.parametrize("now", ["2026-08-31 08:59", "2026-07-08 13:59"])
def test_forecast_does_not_see_after_now(spb, tiny, now):
    full = S.forecast(now, models_dir=tiny, data=spb)
    alt = S.forecast(now, models_dir=tiny, data=_future_garbage(spb, now))
    assert full.records == alt.records
    assert full.explanations == alt.explanations
    assert full.meta == alt.meta


def test_forecast_passes_contract(spb, tiny):
    fc = S.forecast("2026-08-31 08:59", models_dir=tiny, data=spb)
    preds = contract.validate_records(fc.records)
    assert len(preds) == 18 * 2
    assert {p.horizon_min for p in preds} == {60, 120}
    assert {r["ts"] for r in fc.records} == {"2026-08-31T09:00:00+03:00", "2026-08-31T10:00:00+03:00"}
    assert len(fc.explanations) == 36 and all(len(e["reasons"]) == 3 for e in fc.explanations)
    assert fc.meta["model"] == "lgbm_final" and fc.meta["warning"] == S.WARNING   # малютка обучена по 29.09
    assert fc.meta["train_end"] == "2026-09-29"
