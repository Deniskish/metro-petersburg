"""Квантильный LightGBM (этап 4): модель в интерфейсе backtest.Model, абляция групп признаков, объяснения, прогнозы.

Запуск:
  python -m src.model                  # абляция на 4 фолдах (тест май–август) → reports/backtest/model_*.csv
  python -m src.model --reuse          # то же, прогнозы конфигураций берутся из кэша data/interim/model_preds/
  python -m src.model --export-august  # финальная модель из фолда «август» → data/predictions/model_aug.json
                                       # и model_aug_explain.json; демо 31.08; график важности признаков

Модель — три бустера LightGBM (objective="quantile", α = 0,1 / 0,5 / 0,9) на цели
z = log((y + 1) / (f_ref + 1)), f_ref = b4 × уровень (или z = log1p(y) — вариант для сравнения). Веса строк — f_ref:
WAPE взвешен потоком, а без весов цель-отношение учит относительную ошибку малых вестибюлей и крайних часов.
Окна внутри обучения (по суткам метро): C — последние 28 суток, E — последние 14 (E ⊂ C).
  1. Отложенная модель учится на обучении без C, ранняя остановка — по E (тест не участвует).
  2. Конформная поправка q10 и q90 (асимметричная CQR) — по прогнозам отложенной модели на C,
     в ячейках «период суток × h».
  3. Итоговая модель — на всём обучении с числом деревьев из шага 1.
Сентябрь недоступен: панель всегда строится без --final.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import backtest as BT
from src import config, contract
from src import eda_spb as E
from src import features as F

OUT = BT.OUT
CACHE = config.INTERIM / "model_preds"
FINAL_JSON = OUT / "model_final.json"
AUG_OUT = config.PREDICTIONS / "model_aug.json"
EXPLAIN_OUT = config.PREDICTIONS / "model_aug_explain.json"
GAIN_FIG = config.FIGURES / "spb_13_lgbm_gain.png"
MODEL_VERSION = "lgbm_q_ratio_v1"
QUANTILES = BT.QUANTILES
F_REF = "b4_level"
PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": True,
          "force_col_wise": True, "num_threads": 8, "verbose": -1}
MAX_ROUNDS, ES_ROUNDS = 3000, 100
CACHE_VERSION = 1          # поднять при изменении кода модели или признаков: старый кэш не подойдёт
DEMO_STATIONS = ["devyatkino", "ploshchad_lenina", "chernyshevskaya", "vosstaniya", "narvskaya"]
DEMO_HOURS = [8, 9, 10]


# --- Цель -------------------------------------------------------------------------
def to_z(y, f, target: str) -> np.ndarray:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return np.log((y + 1) / (f + 1)) if target == "ratio" else np.log1p(y)


def from_z(z, f, target: str) -> np.ndarray:
    z, f = np.asarray(z, float), np.asarray(f, float)
    y = (f + 1) * np.exp(z) - 1 if target == "ratio" else np.expm1(z)
    return np.maximum(y, 0.0)


def conformal(scores: np.ndarray, miss: float) -> float:
    """Поправка c: доля scores > c — не больше miss, с поправкой на конечную выборку (split conformal)."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - miss)) / n)
    return float(np.quantile(scores, level, method="higher"))


# --- Модель -----------------------------------------------------------------------
@dataclass
class LGBMQuantile:
    name: str
    label: str
    features: list[str]
    target: str = "ratio"                # "ratio" | "log1p"
    params: dict = field(default_factory=lambda: dict(PARAMS))
    max_rounds: int = MAX_ROUNDS
    es_rounds: int = ES_ROUNDS
    es_days: int = 14
    calib_days: int = 28
    cells: tuple = ("band", "h")
    min_cell: int = 300
    boosters: dict = field(default_factory=dict)
    best_iter: dict = field(default_factory=dict)
    corr: pd.DataFrame | None = None     # поправки (lo, hi) по ячейкам, в пространстве z
    corr_all: pd.Series | None = None
    calib_q50: pd.Series | None = None   # q50 отложенной модели на C (индекс — строки обучения)
    info: dict = field(default_factory=dict)

    def _dataset(self, df: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        return lgb.Dataset(df[self.features], to_z(df.y, df[F_REF], self.target),
                           weight=np.maximum(df[F_REF].to_numpy(float), 1.0), reference=reference,
                           free_raw_data=False)

    def _train(self, alpha: float, ds: lgb.Dataset, rounds: int, valid: lgb.Dataset | None = None) -> lgb.Booster:
        params = {**self.params, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
        if valid is None:
            return lgb.train(params, ds, num_boost_round=rounds)
        return lgb.train(params, ds, num_boost_round=rounds, valid_sets=[valid],
                         callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)])

    def windows(self, sday: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Начало окна калибровки C и окна ранней остановки E: последние calib_days и es_days суток обучения."""
        last = sday.max()
        return last - pd.Timedelta(days=self.calib_days - 1), last - pd.Timedelta(days=self.es_days - 1)

    def fit(self, train: pd.DataFrame) -> "LGBMQuantile":
        c0, e0 = self.windows(train.sday)
        hold, cal, es = train[train.sday < c0], train[train.sday >= c0], train[train.sday >= e0]
        ds_hold = self._dataset(hold)
        ds_es = self._dataset(es, reference=ds_hold)
        zc = to_z(cal.y, cal[F_REF], self.target)
        pc = {}
        for a in QUANTILES:
            b = self._train(a, ds_hold, self.max_rounds, ds_es)
            self.best_iter[a] = max(int(b.best_iteration), 1)
            pc[a] = b.predict(cal[self.features], num_iteration=self.best_iter[a])

        scores = pd.DataFrame({"lo": pc[0.1] - zc, "hi": zc - pc[0.9]}, index=cal.index)
        miss = QUANTILES[0]
        keys = [cal[c] for c in self.cells]
        sizes = scores.groupby(keys).size()
        corr = scores.groupby(keys).agg(lambda s: conformal(s.to_numpy(), miss))
        self.corr = corr[sizes >= self.min_cell]
        self.corr_all = scores.apply(lambda s: conformal(s.to_numpy(), miss))
        lo, hi = self._corrections(cal)
        self.calib_q50 = pd.Series(from_z(pc[0.5], cal[F_REF], self.target), index=cal.index)
        self.info = {
            "train_start": str(train.sday.min().date()), "train_end": str(train.sday.max().date()),
            "calib_start": str(c0.date()), "es_start": str(e0.date()), "n_train": len(train), "n_calib": len(cal),
            **{f"best_iter_{int(a * 100)}": self.best_iter[a] for a in QUANTILES},
            "calib_cov_raw": float(np.mean((zc >= pc[0.1]) & (zc <= pc[0.9]))),
            "calib_cov": float(np.mean((zc >= pc[0.1] - lo) & (zc <= pc[0.9] + hi))),
            "corr_lo_all": float(self.corr_all.lo), "corr_hi_all": float(self.corr_all.hi),
        }

        ds_full = self._dataset(train)
        for a in QUANTILES:
            self.boosters[a] = self._train(a, ds_full, self.best_iter[a])
        return self

    def _corrections(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        idx = pd.MultiIndex.from_frame(rows[list(self.cells)])
        lo = self.corr.lo.reindex(idx).fillna(self.corr_all.lo).to_numpy()
        hi = self.corr.hi.reindex(idx).fillna(self.corr_all.hi).to_numpy()
        return lo, hi

    def raw(self, rows: pd.DataFrame) -> dict[float, np.ndarray]:
        """Прогнозы бустеров в пространстве z, без поправки."""
        X = rows[self.features]
        return {a: self.boosters[a].predict(X) for a in QUANTILES}

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        z = self.raw(rows)
        lo, hi = self._corrections(rows)
        f = rows[F_REF].to_numpy(float)
        inv = lambda v: from_z(v, f, self.target)
        qs = np.sort(np.column_stack([inv(z[0.1] - lo), inv(z[0.5]), inv(z[0.9] + hi)]), axis=1)
        raw10, raw90 = np.minimum(inv(z[0.1]), qs[:, 1]), np.maximum(inv(z[0.9]), qs[:, 1])
        small = f < 1                          # нормы почти нет — прогноз равен норме (раздел 6 ТЗ)
        qs[small] = f[small, None]
        raw10[small], raw90[small] = f[small], f[small]
        return pd.DataFrame({"q10": qs[:, 0], "q50": qs[:, 1], "q90": qs[:, 2], "q10_raw": raw10, "q90_raw": raw90},
                            index=rows.index)


def make_model(features: list[str], target: str = "ratio", name: str | None = None,
               label: str | None = None) -> LGBMQuantile:
    return LGBMQuantile(name or f"lgbm_{target}", label or f"LightGBM ({target})", list(features), target)


# --- Объяснения -------------------------------------------------------------------
def contributions(model: LGBMQuantile, rows: pd.DataFrame) -> pd.DataFrame:
    """Вклады признаков в q50 (LightGBM pred_contrib=True), в пространстве z; колонка bias — смещение."""
    c = model.boosters[0.5].predict(rows[model.features], pred_contrib=True)
    return pd.DataFrame(c, columns=[*model.features, "bias"], index=rows.index)


def top_reasons(contrib: pd.DataFrame, values: pd.DataFrame, hours: np.ndarray, places: np.ndarray,
                k: int = 3) -> list[list[dict]]:
    """Топ-k признаков по |вкладу| в каждой строке: признак, текст, вклад в прогноз в % (e^c − 1)."""
    feats = [c for c in contrib.columns if c != "bias"]
    C = contrib[feats].to_numpy()
    top = np.argsort(-np.abs(C), axis=1)[:, :k]
    out = []
    for i in range(len(C)):
        reasons = []
        for j in top[i]:
            name = feats[j]
            reasons.append({"feature": name,
                            "text": F.describe(name, values[name].iloc[i], int(hours[i]), places[i]),
                            "effect_pct": round(float(np.expm1(C[i, j])) * 100, 1)})
        out.append(reasons)
    return out


def explain(model: LGBMQuantile, rows: pd.DataFrame, k: int = 3, names: dict | None = None) -> list[list[dict]]:
    """Топ-3 причины прогноза q50 для строк вестибюлей (names — vestibule_id → название для текста)."""
    places = rows.vestibule_id.map(lambda v: f"вестибюля «{(names or {}).get(v, v)}»").to_numpy()
    return top_reasons(contributions(model, rows), rows, rows.hour.to_numpy(), places, k)


def station_reasons(model: LGBMQuantile, rows: pd.DataFrame, q50: pd.Series, station_names: dict,
                    k: int = 3) -> pd.DataFrame:
    """Причины на уровне станции: вклады вестибюлей с весом доли в q50 станции (лог-линейное приближение),
    значения признаков — средние с тем же весом, входы (F.COUNTS) — сумма, категории — у первого вестибюля."""
    c = contributions(model, rows)
    key = ["station_id", "tau", "h"]
    w = q50 / q50.groupby([rows[c_] for c_ in key]).transform("sum").replace(0, np.nan)
    w = w.fillna(1.0 / rows.groupby(key).vestibule_id.transform("size"))
    feats = list(model.features)
    num = [f for f in feats if f not in F.CATEGORICAL]
    by = [rows[k_] for k_ in key]
    cw = c[feats].mul(w, axis=0).groupby(by, sort=False).sum()
    x = rows[num].astype(float)
    wx = x.mul(w, axis=0).groupby(by, sort=False).sum()
    wn = x.notna().mul(w, axis=0).groupby(by, sort=False).sum()
    vals = wx / wn.where(wn > 0)                                   # среднее с весом, пропуски не тянут к нулю
    counts = [f for f in num if f in F.COUNTS]
    vals[counts] = x[counts].groupby(by, sort=False).sum(min_count=1)
    cats = rows[[*[f for f in feats if f in F.CATEGORICAL], "hour"]].groupby(by, sort=False).first()
    vals = vals.join(cats).reindex(cw.index)
    places = np.array([f"станции «{station_names[s]}»" for s in cw.index.get_level_values("station_id")])
    reasons = top_reasons(cw.assign(bias=0.0), vals, vals.hour.to_numpy(), places, k)
    return pd.DataFrame({"reasons": reasons}, index=cw.index)


# --- Абляция ----------------------------------------------------------------------
def _cache_key(model: LGBMQuantile, fold_list: list[BT.Fold]) -> str:
    spec = {"v": CACHE_VERSION, "features": model.features, "target": model.target, "params": model.params,
            "rounds": [model.max_rounds, model.es_rounds], "windows": [model.es_days, model.calib_days],
            "cells": list(model.cells), "folds": [(f.name, str(f.train_end.date())) for f in fold_list]}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def run_config(rows: pd.DataFrame, model: LGBMQuantile, fold_list: list[BT.Fold],
               reuse: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогнозы модели на тестах фолдов (BT.run) и сведения о калибровке по фолдам; кэш — по признакам и параметрам."""
    key = _cache_key(model, fold_list)
    p_path, i_path = CACHE / f"{model.name}_{key}.parquet", CACHE / f"{model.name}_{key}_info.csv"
    if reuse and p_path.exists() and i_path.exists():
        return pd.read_parquet(p_path), pd.read_csv(i_path)
    infos = []
    pred = BT.run(rows, [model], fold_list, on_fit=lambda f, m: infos.append({"fold": f.name, **m.info}))
    info = pd.DataFrame(infos).assign(model=model.name)
    CACHE.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(p_path)
    info.to_csv(i_path, index=False)
    return pred, info


def _general(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[~pred.is_special]


def fold_delta(prev: pd.DataFrame, cand: pd.DataFrame) -> pd.Series:
    """ΔWAPE = WAPE(prev) − WAPE(cand) по фолдам (> 0 — cand лучше): оба h вместе, особые дни исключены."""
    a, b = _general(prev), _general(cand)
    df = pd.DataFrame({"fold": a.fold.to_numpy(), "y": a.y.to_numpy(), "ea": (a.y - a.q50).abs().to_numpy(),
                       "eb": (b.y - b.q50).abs().to_numpy()})
    s = df.groupby("fold", sort=False).sum()
    return (s.ea - s.eb) / s.y


def compare(base: pd.DataFrame, alt: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """ΔWAPE base → alt (> 0 — alt лучше) с ДИ блочным бутстрепом по дням и p Диболда–Мариано."""
    a, b = _general(base), _general(alt)
    if mask is not None:
        m = mask.loc[a.index].to_numpy()
        a, b = a[m], b[m]
    if a.empty:
        return {"dwape": np.nan, "lo": np.nan, "hi": np.nan, "p_dm": np.nan}
    r = E.compare_forecasts(a.y.to_numpy(), a.q50.to_numpy(), b.q50.to_numpy(), a.sday.to_numpy())
    return {k: r[k] for k in ("dwape", "lo", "hi", "p_dm")}


def summary(pred: pd.DataFrame) -> dict:
    """Метрики шага: WAPE (среднее по h) по срезам, t+1 / t+2, по фолдам; покрытие до и после калибровки."""
    res = BT.evaluate(pred)
    allf = res[res.fold == "все"]
    w = lambda s: allf[allf.slice == s].wape.mean()
    g = _general(pred)
    raw = ((g.y >= g.q10_raw) & (g.y <= g.q90_raw)).groupby(g.h).mean().mean() if "q10_raw" in g else np.nan
    by_fold = res[(res.fold != "все") & (res.slice == "все часы")].groupby("fold").wape.mean()
    return {"wape": w("все часы"), "wape_peaks": w("пики 07–09, 17–19"), "wape_anomaly": w("аномальные часы"),
            "wape_holidays": w("праздники (контроль)"),
            "wape_h1": allf[(allf.slice == "все часы") & (allf.h == 1)].wape.iloc[0],
            "wape_h2": allf[(allf.slice == "все часы") & (allf.h == 2)].wape.iloc[0],
            "coverage_raw": raw, "coverage": allf[allf.slice == "все часы"].coverage.mean(),
            **{f"wape_{f}": by_fold.get(f, np.nan) for f in ("май", "июнь", "июль", "август")}}


def _check_aligned(a: pd.DataFrame, b: pd.DataFrame) -> None:
    keys = ["vestibule_id", "tau", "h", "fold"]
    if not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
        raise RuntimeError("прогнозы конфигураций на разных строках")


def _log(*a) -> None:
    print(*a, flush=True)


def ablation(rows: pd.DataFrame, fold_list: list[BT.Fold], baseline: pd.DataFrame, reuse: bool = False,
             log=_log) -> tuple[pd.DataFrame, dict, dict, list[pd.DataFrame]]:
    """Последовательная абляция: база → + группы в порядке F.ABLATION. Группа остаётся, если средний по фолдам ΔWAPE
    к текущему набору > 0 и положителен хотя бы в 3 фолдах из 4. Затем сравнение целей ratio и log1p на полном наборе."""
    preds, feats_of, infos, steps = {}, {}, [], []

    def fit_cfg(name: str, label: str, feats: list[str], target: str = "ratio") -> pd.DataFrame:
        pred, info = run_config(rows, make_model(feats, target, name, label), fold_list, reuse)
        _check_aligned(baseline, pred)
        preds[name], feats_of[name] = pred, feats
        infos.append(info)
        return pred

    def vs_baseline(pred: pd.DataFrame) -> dict:
        return {f"dbl_{k}": v for k, v in compare(baseline, pred).items()}

    accepted, kept = list(F.GROUPS["база"]), ["база"]
    log("шаг: база")
    current = fit_cfg("lgbm_base", "LightGBM: база", accepted)
    steps.append({"step": "база", "model": "lgbm_base", "n_features": len(accepted), **summary(current),
                  **vs_baseline(current)})
    for i, group in enumerate(F.ABLATION, 1):
        feats = accepted + F.GROUPS[group]
        log(f"шаг: + {group}")
        cand = fit_cfg(f"lgbm_s{i}", f"+ {group}", feats)
        d = fold_delta(current, cand)
        keep = bool(d.mean() > 0 and (d > 0).sum() >= 3)
        row = {"step": f"+ {group}", "model": f"lgbm_s{i}", "n_features": len(feats), "kept": keep, **summary(cand),
               **{f"dprev_{k}": v for k, v in compare(current, cand).items()},
               "dprev_fold_mean": d.mean(), "dprev_folds_pos": int((d > 0).sum()),
               **{f"dprev_{f}": d.get(f, np.nan) for f in ("май", "июнь", "июль", "август")}, **vs_baseline(cand)}
        if group == "особые дни":   # редкие признаки — отдельно ΔWAPE на днях, где они не нулевые
            days = set(rows.loc[rows[F.GROUPS[group]].gt(0).any(axis=1), "sday"])
            row.update({f"dgroup_{k}": v for k, v in compare(current, cand, cand.sday.isin(days)).items()})
            row["group_days"] = ", ".join(sorted(x.strftime("%d.%m") for x in days & set(_general(cand).sday)))
        steps.append(row)
        log(f"  ΔWAPE по фолдам, п. п.: {', '.join(f'{k} {v * 100:+.3f}' for k, v in d.items())} → "
            f"{'оставляем' if keep else 'отбрасываем'}")
        if keep:
            accepted, current = feats, cand
            kept.append(group)

    full = list(F.ALL)
    ratio_name = next((n for n, fs in feats_of.items() if set(fs) == set(full)), None)
    if ratio_name is None:
        log("сравнение целей: ratio на полном наборе")
        fit_cfg("lgbm_full_ratio", "ratio, полный набор", full)
        ratio_name = "lgbm_full_ratio"
    log("сравнение целей: log1p на полном наборе")
    fit_cfg("lgbm_full_log1p", "log1p(y), полный набор", full, "log1p")
    for name, label in ((ratio_name, "цель log((y+1)/(f+1)), полный набор"),
                        ("lgbm_full_log1p", "цель log1p(y), полный набор")):
        steps.append({"step": label, "model": name, "n_features": len(full), **summary(preds[name]),
                      **vs_baseline(preds[name])})
    final = {"features": accepted, "groups": kept, "target": "ratio", "step_model": current.model.iloc[0],
             "ratio_full_model": ratio_name,
             "ratio_vs_log1p": compare(preds["lgbm_full_log1p"], preds[ratio_name])}   # > 0 — ratio лучше
    return pd.DataFrame(steps), preds, final, infos


# --- Отчёт ------------------------------------------------------------------------
def _pct(x: float, nd: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{nd}f} %".replace(".", ",")


def _pp(x: float, nd: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.{nd}f}".replace(".", ",")


def _p(x: float) -> str:
    return "—" if pd.isna(x) else ("< 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ","))


def ablation_markdown(steps: pd.DataFrame, bl: dict) -> str:
    lines = ["| Шаг | WAPE, все часы | пики | аномальные | t+1 / t+2 | покрытие до → после | ΔWAPE к пред. шагу, п. п. [ДИ] "
             "| фолдов с Δ > 0 | решение | к лучшему бейзлайну, п. п. [ДИ], p DM |",
             "|---|---|---|---|---|---|---|---|---|---|",
             f"| {bl['label']} | {_pct(bl['wape'])} | {_pct(bl['wape_peaks'])} | {_pct(bl['wape_anomaly'])} | "
             f"{_pct(bl['wape_h1'])} / {_pct(bl['wape_h2'])} | {_pct(bl['coverage'], 1)} | | | | |"]
    for s in steps.itertuples():
        dprev = (f"{_pp(s.dprev_dwape)} [{_pp(s.dprev_lo)}; {_pp(s.dprev_hi)}]"
                 if hasattr(s, "dprev_dwape") and not pd.isna(getattr(s, "dprev_dwape", np.nan)) else "")
        folds = f"{int(s.dprev_folds_pos)} из 4" if not pd.isna(getattr(s, "dprev_folds_pos", np.nan)) else ""
        decision = "" if pd.isna(s.kept) else ("оставлена" if s.kept else "отброшена")
        if s.step == "база":
            decision = ""
        lines.append(f"| {s.step} | {_pct(s.wape)} | {_pct(s.wape_peaks)} | {_pct(s.wape_anomaly)} | "
                     f"{_pct(s.wape_h1)} / {_pct(s.wape_h2)} | {_pct(s.coverage_raw, 1)} → {_pct(s.coverage, 1)} | "
                     f"{dprev} | {folds} | {decision} | "
                     f"{_pp(s.dbl_dwape)} [{_pp(s.dbl_lo)}; {_pp(s.dbl_hi)}], {_p(s.dbl_p_dm)} |")
    return "\n".join(lines)


# --- Прогнозы для команды ---------------------------------------------------------
def fit_august(panel: BT.Panel, final: dict) -> tuple[LGBMQuantile, pd.DataFrame, pd.DataFrame]:
    """Финальная модель из фолда «август» (обучение до 24.07) и её строки обучения и августа (все открытые часы)."""
    fold = BT.get_fold("август")
    rows = F.add_features(BT.make_rows(panel, for_export=True), panel)
    train, test = BT.split(rows, fold)
    m = make_model(final["features"], final["target"], "lgbm_final", "LightGBM, финальная").fit(train)
    return m, train, test


def export_august(panel: BT.Panel, final: dict) -> tuple[list[dict], list[dict], LGBMQuantile, pd.DataFrame]:
    """Прогнозы финальной модели на август по станциям и топ-3 причины.

    q50 станции — сумма q50 вестибюлей; q10 и q90 — q50 станции × квантили отношения факт / q50 по ячейке
    «группа × период суток × h», калибровка — на прогнозах отложенной модели в окне C (вне выборки)."""
    m, train, test = fit_august(panel, final)
    p = m.predict(test)
    st_test = BT.station_frame(test, p.q50, panel)
    cal_rows = train.loc[m.calib_q50.index]
    st_cal = BT.station_frame(cal_rows, m.calib_q50, panel)
    st_cal = st_cal[(st_cal.n == st_cal.n_open) & st_cal.y.notna() & ~st_cal.is_special]
    q = BT.RatioQuantiles(m.name, m.label, lambda df: df.f, center=False).fit(st_cal).predict(st_test)
    st = st_test.join(q)
    records = BT.station_records(st, panel, MODEL_VERSION)

    names = panel.d.stations.set_index("station_id").name.to_dict()
    sr = station_reasons(m, test, p.q50, names)
    expl = []
    for (sid, tau, h), r in sr.reasons.items():
        local = tau.tz_convert(config.SPB_TZ)
        expl.append({"station_id": sid, "ts": local.isoformat(), "horizon_min": 60 * int(h), "reasons": r})
    expl.sort(key=lambda x: (x["ts"], x["station_id"], x["horizon_min"]))
    return records, expl, m, st.assign(model_q10=st.q10, model_q50=st.q50, model_q90=st.q90)


def demo_0831(panel: BT.Panel, model_st: pd.DataFrame, baseline_name: str, expl: list[dict]) -> pd.DataFrame:
    """31.08, 08–10 ч: факт, норма, бейзлайн и модель по станциям инцидента и по линии."""
    fold = BT.get_fold("август")
    rows = BT.make_rows(panel, for_export=True)
    tr, te = BT.split(rows, fold)
    proto = {m.name: m for m in BT.baseline_ladder()}[baseline_name]
    st_tr = BT.station_frame(tr, proto.point(tr), panel)
    st_tr = st_tr[(st_tr.n == st_tr.n_open) & st_tr.y.notna() & ~st_tr.is_special]
    st_te = BT.station_frame(te, proto.point(te), panel)
    bq = BT.RatioQuantiles(proto.name, proto.label, lambda df: df.f).fit(st_tr).predict(st_te)
    bl = st_te.join(bq)[["station_id", "tau", "h", "hour", "sday", "b", "q10", "q50", "q90"]]
    md = model_st[["station_id", "tau", "h", "model_q10", "model_q50", "model_q90"]]
    df = bl.merge(md, on=["station_id", "tau", "h"])
    df = df[(df.sday == BT.INCIDENT_DAY) & df.hour.isin(DEMO_HOURS)]

    g, ves = panel.d.grid, panel.d.vestibules
    fact = pd.DataFrame(np.where(g.closed, np.nan, g.entries), index=g.ts, columns=ves.station_id.to_numpy())
    fact = fact.T.groupby(level=0).sum(min_count=1).T
    df["fact"] = [fact.at[t, s] for t, s in zip(df.tau, df.station_id)]
    line = df.groupby(["tau", "h", "hour"])[["b", "q50", "model_q50", "fact"]].sum().reset_index()
    line["station_id"] = "вся линия"
    df = pd.concat([df[df.station_id.isin(DEMO_STATIONS)], line], ignore_index=True)
    names = {**panel.d.stations.set_index("station_id").name.to_dict(), "вся линия": "вся линия"}
    df["station"] = df.station_id.map(names)
    reasons = {(e["station_id"], e["ts"], e["horizon_min"]): e["reasons"] for e in expl}
    df["reasons"] = [
        "; ".join(f"{r['text']} ({r['effect_pct']:+.0f} %)" for r in reasons.get(
            (s, t.tz_convert(config.SPB_TZ).isoformat(), 60 * int(h)), []))
        for s, t, h in zip(df.station_id, df.tau, df.h)]
    order = {s: i for i, s in enumerate([*DEMO_STATIONS, "вся линия"])}
    return (df.assign(o=df.station_id.map(order)).sort_values(["o", "hour", "h"])
            [["station", "hour", "h", "fact", "b", "q10", "q50", "q90", "model_q10", "model_q50", "model_q90",
              "reasons"]].rename(columns={"b": "norm", "q10": "baseline_q10", "q50": "baseline_q50",
                                          "q90": "baseline_q90"}).reset_index(drop=True))


# Порядок цветов групп — палитра dataviz (8 слотов, соседние пары различимы при дальтонизме); цвет закреплён
# за группой, а не за рангом. Три цвета светлее 3:1 к фону — поэтому каждый столбец подписан названием признака.
GROUP_COLORS = dict(zip(F.GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))


def gain_figure(m: LGBMQuantile, path=GAIN_FIG, top: int = 25) -> pd.Series:
    """Важность признаков (доля gain) q50-бустера; цвет — группа абляции."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from src import eda
    gain = pd.Series(m.boosters[0.5].feature_importance("gain"), index=m.features)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    show = gain.head(top)[::-1]
    eda.style()
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(show) + 1.6))
    colors = [GROUP_COLORS[F.GROUP_OF[f]] for f in show.index]
    ax.barh(np.arange(len(show)), show.to_numpy() * 100, height=0.62, color=colors)
    ax.set_yticks(np.arange(len(show)), [F.TITLES[f] for f in show.index], fontsize=11)
    for i, v in enumerate(show.to_numpy()):
        if i >= len(show) - 8:                          # подписываем только лидеров
            ax.text(v * 100 + 0.3, i, f"{v * 100:.1f} %".replace(".", ","), va="center", fontsize=10,
                    color=eda.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("доля суммарного gain q50-бустера, %")
    ax.set_title(f"Важность признаков финальной модели (фолд «август», топ-{len(show)} из {len(gain)})", loc="left")
    used = [g for g in F.GROUPS if any(F.GROUP_OF[f] == g for f in show.index)]
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=g) for g in used], loc="lower right")
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return gain


# --- main -------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="брать прогнозы конфигураций из кэша, если признаки те же")
    ap.add_argument("--export-august", action="store_true",
                    help="финальная модель из фолда «август»: прогнозы по контракту, причины, демо 31.08, график")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    panel = BT.load_panel(final=False)

    if args.export_august:
        if not FINAL_JSON.exists():
            raise SystemExit(f"нет {FINAL_JSON.relative_to(config.ROOT)} — сначала python -m src.model")
        final = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        records, expl, m, st = export_august(panel, final)
        contract._write_records(AUG_OUT, records)
        preds = contract.validate_file(AUG_OUT)
        EXPLAIN_OUT.write_text(json.dumps(expl, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{AUG_OUT.relative_to(config.ROOT)}: {len(preds)} записей, контракт OK, "
              f"аномальных {sum(p.is_anomaly for p in preds)}, model_version {preds[0].model_version}")
        print(f"{EXPLAIN_OUT.relative_to(config.ROOT)}: {len(expl)} записей")
        cov = st.dropna(subset=["y"])
        cov = cov[~cov.is_special]
        print(f"покрытие q10–q90 станций в августе (без 31.08): {((cov.y >= cov.q10) & (cov.y <= cov.q90)).mean():.3f}")
        demo = demo_0831(panel, st, final["baseline"], expl)
        demo.round(0).to_csv(OUT / "demo_0831.csv", index=False)
        print(demo.drop(columns="reasons").round(0).to_string())
        gain = gain_figure(m)
        print(gain.round(3).to_string())
        return

    rows = F.add_features(BT.make_rows(panel), panel)
    fold_list = BT.folds()
    ladder = BT.baseline_ladder()
    bl_pred = BT.run(rows, ladder, fold_list)
    bl_res = BT.evaluate(bl_pred)
    best = BT.best_model(bl_res)
    baseline = bl_pred[bl_pred.model == best].reset_index(drop=True)
    print(f"строк: {len(rows):,}; лучший бейзлайн: {best}")

    steps, preds, final, infos = ablation(rows, fold_list, baseline, reuse=args.reuse)
    final["baseline"] = best
    all_pred = pd.concat([bl_pred, *preds.values()], ignore_index=True)
    res = BT.evaluate(all_pred)
    res.round(6).to_csv(OUT / "model_results.csv", index=False)
    BT.evaluate_special(all_pred).round(6).to_csv(OUT / "model_special_days.csv", index=False)
    steps.round(6).to_csv(OUT / "model_ablation.csv", index=False)

    calib = pd.concat(infos, ignore_index=True)
    test_cov = []
    for name, p in preds.items():
        g = _general(p)
        for (fold, h), gg in g.groupby(["fold", "h"], sort=False):
            test_cov.append({"model": name, "fold": fold, "h": h,
                             "test_cov_raw": ((gg.y >= gg.q10_raw) & (gg.y <= gg.q90_raw)).mean(),
                             "test_cov": ((gg.y >= gg.q10) & (gg.y <= gg.q90)).mean()})
    calib = calib.merge(pd.DataFrame(test_cov), on=["model", "fold"], how="left")
    calib.round(5).to_csv(OUT / "model_calibration.csv", index=False)

    bl_sum = {"label": f"{dict((m.name, m.label) for m in ladder)[best]} — лучший бейзлайн", **summary(baseline)}
    md = ablation_markdown(steps, bl_sum)
    (OUT / "model_ablation.md").write_text(md + "\n", encoding="utf-8")
    FINAL_JSON.write_text(json.dumps(final, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(md)
    print(f"финальный набор: {final['groups']}; ratio против log1p на полном наборе: {final['ratio_vs_log1p']}")


if __name__ == "__main__":
    main()
