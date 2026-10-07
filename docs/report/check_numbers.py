"""Сверка ключевых чисел отчёта с источниками.

Каждое значение считается из reports/backtest/*.csv|json, data/features_screening.csv или logs/clean_spb.log,
форматируется так же, как в тексте («5,51 %», «+1,89 [+1,57; +2,24]»), и ищется в docs/report/report.md рядом
с ключевым словом контекста (в пределах ±400 символов). При любом расхождении — код возврата 1.

Запуск из корня: python docs/report/check_numbers.py
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "reports" / "backtest"
REPORT = ROOT / "docs" / "report" / "report.md"
WINDOW = 400


def _r(x: float, nd: int) -> str:
    """Арифметическое округление (0,195 → 0,20) десятичной записи числа, а не его двоичного приближения."""
    q = Decimal(1).scaleb(-nd)
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    return str(d.quantize(q, rounding=ROUND_HALF_UP)).replace(".", ",")


def pct(x: float, nd: int = 2) -> str:
    return _r(Decimal(repr(float(x))) * 100, nd) + " %"


def pp(x: float) -> str:
    v = Decimal(repr(float(x))) * 100
    return ("−" if x < 0 else "+") + _r(abs(v), 2)


def ci(d: float, lo: float, hi: float) -> str:
    return f"{pp(d)} [{pp(lo)}; {pp(hi)}]"


def f2(x: float) -> str:
    return _r(x, 2)


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def wape(res: pd.DataFrame, model: str, sl: str = "все часы", h: int | None = None, col: str = "wape") -> float:
    r = res[(res.model == model) & (res.fold == "все") & (res.slice == sl)]
    return float(r[r.h == h][col].iloc[0]) if h else float(r[col].mean())


def build_checks() -> list[tuple[str, str, str]]:
    """(что проверяем, ожидаемая строка, слово контекста рядом)."""
    cfg = json.loads((B / "model_final.json").read_text(encoding="utf-8"))
    final = cfg["step_model"]
    best = cfg["baseline"]
    res = pd.read_csv(B / "model_results.csv")
    abl = pd.read_csv(B / "model_ablation.csv").set_index("step")
    loo = pd.read_csv(B / "model_loo.csv").set_index("group")
    fres = pd.read_csv(B / "final_results.csv")
    fcmp = pd.read_csv(B / "final_compare.csv").set_index("period")
    fcal = pd.read_csv(B / "final_calibration.csv").set_index("h")
    mcal = pd.read_csv(B / "model_calibration.csv")
    flags = pd.read_csv(B / "model_flags.csv")
    fflags = pd.read_csv(B / "final_flags.csv")
    scr = pd.read_csv(ROOT / "data" / "features_screening.csv")
    log = (ROOT / "logs" / "clean_spb.log").read_text(encoding="utf-8")

    c = []
    # --- бэктест: финальная модель и лучший бейзлайн
    c += [("бэктест: WAPE модели", pct(wape(res, final)), "финальн"),
          ("бэктест: WAPE модели t+1", pct(wape(res, final, h=1)), "t+1"),
          ("бэктест: WAPE модели t+2", pct(wape(res, final, h=2)), "t+2"),
          ("бэктест: пики модели", pct(wape(res, final, "пики 07–09, 17–19")), "пик"),
          ("бэктест: аномальные часы модели", pct(wape(res, final, "аномальные часы")), "аномальн"),
          ("бэктест: покрытие модели", pct(wape(res, final, col="coverage"), 1), "Покрыти"),
          ("бэктест: WAPE лучшего бейзлайна", pct(wape(res, best)), "бейзлайн"),
          ("бэктест: бейзлайн t+1 / t+2", f"{pct(wape(res, best, h=1))} / {pct(wape(res, best, h=2))}", "уровень × r(t)"),
          ("бэктест: аномальные часы бейзлайна", pct(wape(res, best, "аномальные часы")), "аномальн"),
          ("бэктест: покрытие бейзлайна", pct(wape(res, best, col="coverage"), 1), "уровень × r(t)")]
    n_test = res[(res.model == final) & (res.fold == "все") & (res.slice == "все часы")].n.sum()
    c.append(("бэктест: число тестовых строк", num(int(n_test)), "строк"))
    for m, label in (("seasonal_naive", "Сезонный наивный"), ("b4", "Норма b"), ("b8", "8 недел"),
                     ("b4_level", "уровень"), ("b4_x_r", "b × r(t)"), ("b4_x_ewm2", "сглаженное r")):
        c.append((f"лестница: {label}", pct(wape(res, m)), label))
    # --- абляция и цель
    dbl = abl.loc["+ погода"]
    c.append(("ΔWAPE модели к бейзлайну", ci(dbl.dbl_dwape, dbl.dbl_lo, dbl.dbl_hi), "бейзлайн"))
    for step in ("+ ядро динамики", "+ кросс", "+ скорость и ускорение", "+ вчера", "+ погода", "+ особые дни"):
        r = abl.loc[step]
        c.append((f"абляция {step}: ΔWAPE", ci(r.dprev_dwape, r.dprev_lo, r.dprev_hi), step.lstrip("+ ").split()[0]))
        c.append((f"абляция {step}: WAPE", pct(r.wape), step.lstrip("+ ").split()[0]))
    sd = abl.loc["+ особые дни"]
    c.append(("особые дни на своих днях", f"{pp(sd.dgroup_dwape)} п. п. [{f2(sd.dgroup_lo * 100)}; {pp(sd.dgroup_hi)}]",
              "8.05"))
    c.append(("покрытие до поправки, бэктест", pct(abl.loc["+ погода"].coverage_raw, 1), "поправк"))
    rl = cfg["ratio_vs_log1p"]
    c.append(("цель-отношение против log1p", ci(rl["dwape"], rl["lo"], rl["hi"]), "log1p"))
    c.append(("WAPE цели log1p", pct(abl.loc["цель log1p(y), полный набор"].wape), "log1p"))
    # --- «минус одна группа»
    for g in loo.index:
        r = loo.loc[g]
        c.append((f"минус «{g}»: потеря", ci(r.loss_dwape, r.loss_lo, r.loss_hi), g))
    # --- смесь
    t = cfg["blend"]["test"]
    c += [("смесь: WAPE", pct(t["wape_blend"]), "Смесь"),
          ("смесь: аномальные часы", pct(t["wape_anomaly_blend"]), "аномальн"),
          ("смесь: ΔWAPE все часы, ДИ", f"[{pp(t['all']['lo'])}; {pp(t['all']['hi'])}]", "0,064"),
          ("смесь: ΔWAPE аномальные, ДИ", ci(t["anomaly"]["dwape"], t["anomaly"]["lo"], t["anomaly"]["hi"]), "0,046")]
    # --- калибровка по фолдам
    cv = mcal[mcal.model == final].groupby("fold").test_cov.mean()
    c.append(("покрытие по фолдам, диапазон", f"{pct(cv.min(), 1)[:-2]}–{pct(cv.max(), 1)}", "фолдам"))
    # --- флаг
    fl = flags[(flags.model == "модель") & (flags.level == "вестибюли") & (flags.h == "оба") & (flags.rule == "порог p80")].iloc[0]
    c.append(("флаг p80, бэктест: P / R / F1", f"{f2(fl.precision)} / {f2(fl.recall)} / {f2(fl.f1)}", "p80"))
    ff = fflags[(fflags.model == "lgbm_final") & (fflags.level == "вестибюли") & (fflags.h.astype(str) == "оба")
                & (fflags.rule == "порог p80")].iloc[0]
    c.append(("флаг p80, сентябрь: P / R / F1", f"precision {f2(ff.precision)}, recall {f2(ff.recall)}, F1 {f2(ff.f1)}",
              "Модель"))
    # --- финальный тест
    fm = "lgbm_final"
    c += [("финал: WAPE модели", pct(wape(fres, fm)), "LightGBM"),
          ("финал: модель t+1 / t+2", f"{pct(wape(fres, fm, h=1))} / {pct(wape(fres, fm, h=2))}", "LightGBM"),
          ("финал: пики модели", pct(wape(fres, fm, "пики 07–09, 17–19")), "пик"),
          ("финал: аномальные часы модели", pct(wape(fres, fm, "аномальные часы")), "аномальн"),
          ("финал: покрытие модели", pct(wape(fres, fm, col="coverage"), 1), "покры"),
          ("финал: WAPE бейзлайна", pct(wape(fres, best)), "бейзлайн"),
          ("финал: сезонный наивный", pct(wape(fres, "seasonal_naive")), "Сезонный"),
          ("финал: норма b", pct(wape(fres, "b4")), "Норма b")]
    for period, ctx in (("весь тест", "сентябр"), ("01.09–14.09", "01–14.09"), ("15.09–29.09", "15–29.09"),
                        ("аномальные часы", "аномальн")):
        r = fcmp.loc[period]
        c.append((f"финал ΔWAPE: {period}", ci(r.dwape, r.lo, r.hi), ctx))
        c.append((f"финал WAPE модели: {period}", pct(r.wape_alt), ctx))
        c.append((f"финал WAPE бейзлайна: {period}", pct(r.wape_base), ctx))
    c += [("финал: покрытие t+1 / t+2", f"{pct(fcal.loc[1].test_cov, 1)} на t+1 и {pct(fcal.loc[2].test_cov, 1)} на t+2", "Калибровка"),
          ("финал: покрытие до поправки", f"{pct(fcal.loc[1].test_cov_raw, 1)} и {pct(fcal.loc[2].test_cov_raw, 1)}", "До поправки")]
    n_sep = fres[(fres.model == fm) & (fres.fold == "все") & (fres.slice == "все часы")].n.sum()
    c.append(("финал: число строк", num(int(n_sep)), "строк"))
    # --- отбор: share_dev
    for h in (1, 2):
        r = scr[(scr.candidate == "share_dev") & (scr.h == h)].iloc[0]
        c.append((f"share_dev, t+{h}", ci(r.dwape, r.dwape_lo, r.dwape_hi), "share_dev" if h == 1 else "t+2"))
    # --- журнал очистки
    rows = int(re.search(r"строк: ([\d,]+);", log).group(1).replace(",", ""))
    total = int(re.search(r"сумма входов: ([\d,]+)", log).group(1).replace(",", ""))
    c += [("очистка: строк", num(rows), "строк"), ("очистка: сумма входов", num(total), "вход")]
    for label, ctx in (("без флага", "без флага"), ("метро закрыто (01–04)", "метро закрыто"),
                       ("вестибюль закрыт: режим", "по режиму"), ("инцидент", "инцидент")):
        n = int(re.search(rf"{re.escape(label)}\s+(\d+)", log).group(1))
        c.append((f"флаг «{label}»: строк", num(n), ctx))
    return c


def main() -> int:
    text = REPORT.read_text(encoding="utf-8").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    fails = 0
    for what, value, ctx in build_checks():
        variants = {value, value.replace(" [", " п. п. [", 1)}
        pos = [m.start() for v in variants for m in re.finditer(re.escape(v), text)]
        near = any(ctx.lower() in text[max(0, p - WINDOW):p + WINDOW].lower() for p in pos)
        status = "OK" if near else ("НЕТ РЯДОМ С КОНТЕКСТОМ" if pos else "НЕТ В ОТЧЁТЕ")
        if status != "OK":
            fails += 1
        print(f"{'OK  ' if status == 'OK' else 'FAIL'} {what}: «{value}»" + ("" if status == "OK" else f" — {status} «{ctx}»"))
    total = len(build_checks())
    print(f"\nпроверок: {total}, расхождений: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
