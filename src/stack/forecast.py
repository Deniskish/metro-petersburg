"""Стекинг в продакшн-прогнозе: serve.forecast(now, model="stack") (этап 7, ч. 3).

**Момент прогноза.** t = floor_30мин(now + 1 мин): получасие hh:00–hh:29 закрыто в hh:29, так же как час в serve.origin
закрыт в hh:59. now = 08:59 → t = 09:00; now = 09:29 → t = 09:30. Последний закрытый час t0 = floor_hour(t) − 1 ч —
тот же, что serve.origin(now).

**Выход.** Записи по контракту: станция × получасовой слот [t + 30(k − 1), t + 30k) × горизонт 30k, k = 1…4 —
то, что оценивалось на этапе 7 (docs/stack_findings.md). Причины — от часовой LightGBM (B1) на час слота: GRU
и персистентность объяснений не дают.

**Только данные до now.**
- 15-минутные четверти с ts ≥ t обнуляются (cut_quarters). Норма четвертей текущего, ещё не закрытого часа считается
  без флагов, выведенных из полного часа (is_source_mismatch, is_slot_closed): они видят будущие четверти.
- Часовые данные режутся serve.cut_data(d, t0).
- Профиль и поправка источника — по суткам строго до D (intrahour.past_profile, source_scale); квантили остатка B2 —
  по месяцам обучения выбранной GRU.

**Честный выбор моделей по дате.**
- B1 — serve.choose_model: самая свежая LightGBM с train_end + 7 суток < now, иначе lgbm_final с предупреждением.
- B3 — так же среди models/stack_gru_*: нет модели вне выборки — stack_gru_final с предупреждением.
- Мета-модель (веса) подобрана на мае и июле: для исторических now в эти месяцы она внутри выборки — это в meta.

**Когда стекинга нет** — Unavailable с причиной, serve.forecast уходит на часовую LightGBM с пометкой в meta:
нет 15-минутных данных за t (они есть за февраль, май, июль, сентябрь), окно 4 ч выходит за блок месяца, t или все слоты
вне часов 06–00, необычные сутки (праздник, событие: стекинг на них не учился и не проверялся), нет файлов стекинга,
нет ни одной полной станции.
"""
import functools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as BT
from src import config
from src import eda_spb as E
from src import external_adapter as A
from src import features as F
from src import intrahour as I
from src import model as M
from src import serve as S
from src.stack import base as B
from src.stack import data as D
from src.stack import evaluate as V
from src.stack import meta as MT
from src.stack import nn as NN

GRU_PREFIX = "stack_gru_"
GRU_FINAL = "stack_gru_final"
PARAMS_JSON = config.ROOT / "reports" / "stack" / "stack_params.json"
GRU_WARNING = "GRU видела этот период: прогноз не вне выборки"
META_WARNING = "веса стекинга подобраны на мае и июле: для этого месяца мета-модель внутри выборки"


class Unavailable(Exception):
    """Стекинг для этого момента недоступен — причина для meta["fallback"]."""


def origin(now) -> pd.Timestamp:
    """t (местное время): начало получасия прогноза."""
    return (S._local(now) + pd.Timedelta(minutes=1)).floor("30min")


# --- Выбор GRU по дате -----------------------------------------------------------------
def gru_bundles(models_dir: Path = M.MODELS) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8"))["info"]
            for p in sorted(Path(models_dir).glob(f"{GRU_PREFIX}*/meta.json"))}


def choose_gru(now, models_dir: Path = M.MODELS) -> tuple[str, bool, str | None]:
    """(GRU, вне выборки ли, предупреждение): самая свежая с train_end + 7 суток < now; нет — stack_gru_final."""
    now = S._local(now)
    found = gru_bundles(models_dir)
    if not found:
        raise Unavailable(f"нет моделей GRU в {models_dir} — python -m src.stack --fit")
    oos = {n: pd.Timestamp(i["train_end"]) + S.GAP < now for n, i in found.items()}
    ok = [n for n in found if oos[n] and n != GRU_FINAL]
    if ok:
        return max(ok, key=lambda n: found[n]["train_end"]), True, None
    if GRU_FINAL in found:
        return GRU_FINAL, oos[GRU_FINAL], None if oos[GRU_FINAL] else GRU_WARNING
    raise Unavailable("нет GRU, обученной до этого момента, и нет stack_gru_final — python -m src.stack --train-final")


@functools.lru_cache(maxsize=4)
def _gru(path: str) -> NN.GRUQuantile:
    return NN.GRUQuantile.load(Path(path))


# --- 15-минутные данные ------------------------------------------------------------------
class Context:
    """Всё, что зависит только от 15-минутного файла: таблица часов, сетка четвертей, профиль и поправка источника
    по прошлому, строки этапа 7 (для квантилей остатка B2 по месяцам обучения GRU)."""

    def __init__(self, hours: pd.DataFrame):
        self.hours = hours
        self.sd = D.build(hours)
        prof, sv = I.past_profile(hours), I.source_scale(hours)
        key = pd.MultiIndex.from_frame(hours[["vestibule_id", "ts_utc"]])
        self.profile = prof[I.PCOLS].set_axis(key)
        self.s_v = pd.Series(sv.to_numpy(), index=key)
        self._rq: dict[str, dict] = {}

    def residual_quantiles(self, train_end: str) -> dict:
        """Квантили остатка B2 по строкам суток ≤ train_end (те же месяцы, на которых училась выбранная GRU)."""
        if train_end not in self._rq:
            tr = self.sd.rows[self.sd.rows.sday <= pd.Timestamp(train_end)]
            self._rq[train_end] = B.fit_residual_quantiles(tr, B.b2_point(self.sd.q, tr))
        return self._rq[train_end]


@functools.lru_cache(maxsize=1)
def _default_context() -> Context:
    if not config.SPB_15MIN.exists():
        raise Unavailable("нет 15-минутных данных — python -m src.clean_spb_15min")
    return Context(I.load_hours())


def cut_quarters(q: D.Quarters, t_utc: pd.Timestamp) -> D.Quarters:
    """Сетка четвертей на момент t: четверти с ts ≥ t — пропуск."""
    after = (q.ts >= t_utc)[:, None]
    return replace(q, y=np.where(after, np.nan, q.y), norm=np.where(after, np.nan, q.norm))


# --- Строки прогноза ------------------------------------------------------------------------
def slot_rows(t_utc: pd.Timestamp, hourly: pd.DataFrame, ctx: Context, calendar: pd.DataFrame) -> pd.DataFrame:
    """(вестибюль, t, k) без факта: слот в часах 06–00, вестибюль открыт по расписанию (есть строка часовой модели),
    b4 ≥ 20, есть профиль и s_v; только обычные сутки."""
    t0 = D.last_closed_hour(pd.Series([t_utc]))[0]
    parts = []
    for k in range(1, D.K + 1):
        slot = t_utc + (k - 1) * D.SLOT
        hour_ts = slot.floor("h")
        h = int((hour_ts - t0) / pd.Timedelta(hours=1))
        r = hourly[(hourly.tau == hour_ts) & (hourly.h == h)]
        parts.append(r.assign(k=k, horizon=30 * k, slot=slot, half=int(slot.minute == 30), hour_ts=hour_ts, h_hour=h))
    rows = pd.concat(parts, ignore_index=True)
    if D.local_hour(pd.Series([t_utc]))[0] not in D.HOURS:
        raise Unavailable(f"момент {t_utc.tz_convert(config.SPB_TZ):%H:%M} вне часов 06–00")
    rows = rows[rows.hour.isin(D.HOURS)]
    regular = calendar.set_index("date").is_regular.reindex(rows.sday).eq(True).to_numpy()
    if len(rows) and not regular.any():
        raise Unavailable("необычные сутки (праздник, перенос или событие): стекинг на них не учился и не проверялся")
    rows = rows[regular]
    key = pd.MultiIndex.from_frame(rows[["vestibule_id", "hour_ts"]])
    p = ctx.profile.reindex(key).to_numpy()
    rows["s_v"] = ctx.s_v.reindex(key).to_numpy()
    half = rows.half.to_numpy()
    rows["p_half"] = np.where(half == 0, p[:, 0] + p[:, 1], p[:, 2] + p[:, 3])
    rows = rows[(rows.b4 >= E.B_MIN) & rows.p_half.notna() & rows.s_v.notna()].copy()
    rows["n"] = rows.b4 * rows.s_v * rows.p_half
    rows["t"] = t_utc
    rows["t0"] = t0
    rows["minute"] = t_utc.minute
    rows["band4"] = rows.hour.map(D.BAND_OF)
    rows["z"] = np.nan
    rows["y"] = np.nan
    if rows.empty:
        raise Unavailable("нет слотов прогноза: все слоты вне часов 06–00 или без нормы и профиля")
    return rows.reset_index(drop=True)


def patch_current_hour(q: D.Quarters, t_utc: pd.Timestamp, hourly: pd.DataFrame, ctx: Context) -> D.Quarters:
    """Норма четвертей текущего, незакрытого часа (t = hh:30): b4 × s_v × p_j без флагов полного часа."""
    hour_ts = t_utc.floor("h")
    if hour_ts == t_utc:
        return q
    cur = hourly[(hourly.tau == hour_ts) & (hourly.h == 1)].set_index("vestibule_id").b4
    norm = q.norm.copy()
    for j in range(int((t_utc - hour_ts) / D.QUARTER)):
        i = q.index_of(pd.Series([hour_ts + j * D.QUARTER]))[0]
        if i < 0:
            continue
        for v, ves in enumerate(q.vestibules):
            b4 = cur.get(ves, np.nan)
            key = (ves, hour_ts)
            p = ctx.profile.loc[key, I.PCOLS[j]] if key in ctx.profile.index else np.nan
            s = ctx.s_v.get(key, np.nan)
            ok = np.isfinite(b4) and b4 >= E.B_MIN and np.isfinite(p) and np.isfinite(s) and np.isfinite(q.y[i, v])
            norm[i, v] = b4 * s * p if ok else np.nan
    return replace(q, norm=norm)


# --- Станции ------------------------------------------------------------------------------
def station_frame(df: pd.DataFrame, panel: BT.Panel) -> pd.DataFrame:
    """Суммы вестибюлей в станцию по (станция, k); станция полная, если есть все вестибюли, открытые в час слота."""
    d = df.assign(q50=B.to_y(df, "stack").q50.to_numpy())
    st = d.groupby(["station_id", "k"]).agg(
        n=("n", "sum"), q50=("q50", "sum"), cnt=("vestibule_id", "size"), slot=("slot", "first"),
        hour_ts=("hour_ts", "first"), h_hour=("h_hour", "first"), hour=("hour", "first"), band4=("band4", "first"),
        group=("group", "first"), horizon=("horizon", "first")).reset_index()
    g, ves = panel.d.grid, panel.d.vestibules
    open_ = pd.DataFrame(~g.closed, index=g.ts, columns=ves.station_id.to_numpy()).T.groupby(level=0).sum().T.stack()
    st["n_open"] = open_.reindex(pd.MultiIndex.from_arrays([st.hour_ts, st.station_id])).to_numpy()
    return st[st.cnt == st.n_open].reset_index(drop=True)


def _weights(params: dict) -> dict[str, dict]:
    """Веса базовых моделей: {модель: {q10: {30: w, …}, …}} из stack_params.json."""
    w = params["weights"]
    out = {name: {} for name in w["names"]}
    for q, cells in w["w_k"].items():
        for k, ws in cells.items():
            for name, x in zip(w["names"], ws):
                out[name].setdefault(f"q{q}", {})[str(30 * int(k))] = x
    return out


# --- Прогноз ------------------------------------------------------------------------------
def frame(now, models_dir: Path = M.MODELS, data: E.SpbData | None = None, hours: pd.DataFrame | None = None,
          weather_source: str = "auto", weather_provider=None):
    """Прогноз по вестибюлям: строки с z базовых моделей и стекинга + всё, что нужно для записей и meta."""
    if not PARAMS_JSON.exists():
        raise Unavailable("нет reports/stack/stack_params.json — python -m src.stack --fit")
    params = MT.load_params(PARAMS_JSON)
    t = origin(now)
    t_utc = t.tz_localize(config.SPB_TZ).tz_convert("UTC")
    t0_utc = t_utc.floor("h") - pd.Timedelta(hours=1)
    d = data if data is not None else S._data()
    if t0_utc not in d.grid.ts or t < BT.TRAIN_START:
        raise ValueError(f"now = {now}: вне периода данных ({BT.TRAIN_START:%d.%m.%Y} – {d.grid.local[-1]:%d.%m.%Y})")
    ctx = Context(hours) if hours is not None else _default_context()
    q = ctx.sd.q
    if q.index_of(pd.Series([t_utc]))[0] < 0 or not q.window_ok(pd.Series([t_utc]))[0]:
        raise Unavailable("нет 15-минутных данных за 4 ч до момента прогноза (они есть за февраль, май, июль и сентябрь)")

    b1_name, b1_oos, b1_warn = S.choose_model(now, models_dir, "auto")
    gru_name, gru_oos, gru_warn = choose_gru(now, models_dir)
    m, _, b1_meta = S._bundle(str(Path(models_dir) / b1_name))
    gru = _gru(str(Path(models_dir) / gru_name))

    live = S.is_live(now)
    source = ("yandex" if live else "openmeteo") if weather_source == "auto" else weather_source
    weather_fc, weather = A.weather_table(d.forecast, t0_utc, source, weather_provider)
    d = replace(d, forecast=weather_fc)
    panel = BT.build_panel(S.cut_data(d, t0_utc), final=True)
    panel = replace(panel, thresholds=pd.DataFrame(b1_meta["thresholds"]))
    hourly = BT.make_rows(panel, horizons=B.H_HOURLY, for_export=True)
    hourly = F.add_features(hourly[hourly.t == t0_utc], panel)

    rows = slot_rows(t_utc, hourly, ctx, d.calendar)
    qc = patch_current_hour(cut_quarters(q, t_utc), t_utc, hourly, ctx)

    # B1: часовая модель → слот
    used = hourly.set_index(["vestibule_id", "tau", "h"]).loc[
        pd.MultiIndex.from_frame(rows[["vestibule_id", "hour_ts", "h_hour"]]).unique()].reset_index()
    p1 = m.predict(used)
    hourly_pred = pd.DataFrame({"vestibule_id": used.vestibule_id, "t0": used.t, "hour_ts": used.tau, "h_hour": used.h,
                                "m10": p1.q10.to_numpy(), "m50": p1.q50.to_numpy(), "m90": p1.q90.to_numpy(),
                                "b1_model": b1_name})
    df = rows.join(B.b1_slot(rows, hourly_pred))
    # B2: персистентность, квантили остатка — по месяцам обучения GRU
    rq = ctx.residual_quantiles(gru.info["train_end"])
    z2 = B.apply_residual_quantiles(df, B.b2_point(qc, df), rq)
    df = df.join(pd.DataFrame(z2, columns=B.cols("b2"), index=df.index))
    # B3: GRU
    s = NN.samples(qc, df)
    df = df.join(NN.to_rows(s, gru.predict(s), df))
    df = df.join(MT.predict(df, params))

    month = t.month
    meta_oos = month not in MT.FIT_MONTHS
    info = {"t": t_utc.tz_convert(config.SPB_TZ).isoformat(), "params": params, "b1": (b1_name, b1_oos, b1_warn, b1_meta),
            "gru": (gru_name, gru_oos, gru_warn, gru.info), "meta_oos": meta_oos, "weather": weather, "live": live,
            "hourly": hourly, "used": used, "p1": p1, "panel": panel, "model": m}
    return df, info


def forecast(now, models_dir: Path = M.MODELS, data: E.SpbData | None = None, hours: pd.DataFrame | None = None,
             weather_source: str = "auto", weather_provider=None, railway_provider=None,
             events: list[A.LocatedEvent] | None = None) -> S.Forecast:
    """Записи по контракту (станция × получасовой слот × горизонт 30/60/90/120), причины, meta."""
    df, info = frame(now, models_dir, data, hours, weather_source, weather_provider)
    params, panel = info["params"], info["panel"]
    b1_name, b1_oos, b1_warn, b1_meta = info["b1"]
    gru_name, gru_oos, gru_warn, gru_info = info["gru"]

    st = station_frame(df, panel)
    if st.empty:
        raise Unavailable("нет ни одной полной станции")
    st = st.join(V.station_interval(st, params["station_quantiles"]).drop(columns="q50"))
    version = f"{params['version']}/{b1_name}+{gru_name}"
    records = V.station_records(st, b1_meta["anomaly_rule"], pd.DataFrame(b1_meta["thresholds"]), version)

    # причины — от часовой LightGBM на час слота (две половины часа делят причины)
    names = panel.d.stations.set_index("station_id").name.to_dict()
    sr = M.station_reasons(M._lgbm(info["model"]), info["used"], info["p1"].q50, names).reasons
    expl = []
    for r in st.itertuples():
        expl.append({"station_id": r.station_id, "ts": r.slot.tz_convert(config.SPB_TZ).isoformat(),
                     "horizon_min": int(r.horizon), "reasons": sr[(r.station_id, r.hour_ts, r.h_hour)]})
    expl.sort(key=lambda x: (x["ts"], x["station_id"], x["horizon_min"]))
    context = A.add_context(expl, railway_provider, events, info["live"])

    w = _weights(params)
    base = [{"name": "B1", "kind": "lgbm", "bundle": b1_name, "train_start": b1_meta["train_start"],
             "train_end": b1_meta["train_end"], "out_of_sample": b1_oos, "weights": w["b1"]},
            {"name": "B2", "kind": "persistence", "quarters": B.PERSIST_QUARTERS,
             "residual_quantiles_until": gru_info["train_end"], "weights": w["b2"]},
            {"name": "B3", "kind": "gru", "bundle": gru_name, "train_start": gru_info["train_start"],
             "train_end": gru_info["train_end"], "out_of_sample": gru_oos, "weights": w["b3"]}]
    warnings = [x for x in (b1_warn, gru_warn, None if info["meta_oos"] else META_WARNING) if x]
    t = pd.Timestamp(info["t"])
    meta = {"model": "stack", "requested_model": "stack", "model_version": params["version"],
            "base_models": base,
            "meta_model": {"params": str(PARAMS_JSON.relative_to(config.ROOT)), "sha256": params["sha256"],
                           "variant": params["variant_label"], "fit_months": params["fit_months"],
                           "out_of_sample": info["meta_oos"]},
            "train_start": min(b1_meta["train_start"], gru_info["train_start"]),
            "train_end": max(b1_meta["train_end"], gru_info["train_end"]),
            "out_of_sample": bool(b1_oos and gru_oos), "warning": "; ".join(warnings) or None,
            "now": str(S._local(now)), "t": info["t"],
            "t0": (t.floor("h") - pd.Timedelta(hours=1)).isoformat(),
            "slot_minutes": 30, "horizons_min": list(D.HORIZONS), "explanations": "lgbm (B1), час слота",
            "stations": int(st.station_id.nunique()), "config_sha256": b1_meta.get("config_sha256"),
            "weather": {"requested": weather_source, **info["weather"],
                        "features": S._weather_features(info["used"])},
            "context": context}
    return S.Forecast(records, expl, meta)
