# Прогноз пассажиропотока 1 линии метро СПб

ML-часть проекта: прогноз входов на 18 станций 1 линии на 1 и 2 часа вперёд с интервалом q10–q90, флагом аномалии
и тремя причинами прогноза для LLM-слоя.

- Модель — [docs/model_card.md](docs/model_card.md).
- Бэктест и финальный тест — [docs/backtest.md](docs/backtest.md).
- Ход работ — [docs/progress.md](docs/progress.md).
- Контракт выхода — [src/contract.py](src/contract.py) и [data/predictions/contract.schema.json](data/predictions/contract.schema.json).

## Как получить прогноз

### 1. Установка

Нужен Python 3.12. Файлы организаторов (`Пассажиропоток 2026 Линия 1.xlsx` и др.) кладутся в `data/raw/spb/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --only weather_spb calendar   # погода Open-Meteo и производственный календарь
python -m src.clean_spb                              # → data/interim/
```

### 2. Модели

Модели не хранятся в git, их нужно собрать (около 5 минут). Конфигурация — замороженная
[reports/backtest/model_final.json](reports/backtest/model_final.json); команды проверяют её хэш.

```bash
python -m src.model --save-folds        # models/lgbm_fold_2026-05 … 2026-08: обучение до начала каждого месяца
python -m src.serve --train --holdout   # models/lgbm_holdout: обучение по 24.08 (та же модель, что в финальном тесте)
python -m src.serve --train             # models/lgbm_final: все данные по 29.09
```

Сам финальный тест (`python -m src.model --final`) уже проведён и повторно не запускается.

### 3. Прогноз

```bash
python -m src.serve --now "2026-08-31 08:59"
python -m src.serve --now "2026-08-31 08:59" --stations devyatkino,narvskaya --json forecast.json
```

Из Python (например, в Streamlit):

```python
from src import serve

fc = serve.forecast("2026-08-31 08:59")     # данные только до now; модель выбирается автоматически
fc.records        # прогноз по контракту: 18 станций × горизонты 60 и 120 мин
fc.explanations   # топ-3 причины на каждую запись
fc.meta           # какая модель, до какой даты обучена, вне выборки ли прогноз
serve.table(fc)   # всё одной таблицей
```

**Момент прогноза.** Час hh считается закрытым в hh:59. `now = 08:59` — это данные по 08 ч включительно и слоты 09:00
(горизонт 60 мин) и 10:00 (120 мин). Поток после этого часа не используется — это проверяет тест.

**Режим повтора.** Можно задать любой исторический `now` с 09.02.2026. По умолчанию (`--model auto`) берётся
самая свежая модель, у которой последние сутки обучения + 7 суток раньше `now`. Так прогноз на прошлую дату делает
модель, обученная до неё:

| `now` | Модель | Обучена по |
|---|---|---|
| 30.04 – 30.05 | `lgbm_fold_2026-05` | 23.04 |
| 31.05 – 29.06 | `lgbm_fold_2026-06` | 24.05 |
| 30.06 – 30.07 | `lgbm_fold_2026-07` | 23.06 |
| 31.07 – 30.08 | `lgbm_fold_2026-08` | 24.07 |
| 31.08 – 05.10 | `lgbm_holdout` | 24.08 |
| с 06.10 | `lgbm_final` | 29.09 |

Раньше 30.04 подходящей модели нет. Тогда используется `lgbm_final` с предупреждением «модель видела этот период»
в `meta.warning`: её веса обучены на более поздних данных. Модель можно задать и явно: `--model lgbm_final`.

### 4. Формат выхода

`records` — список записей контракта ([src/contract.py](src/contract.py)). Пример — `--now "2026-08-31 08:59"`,
Девяткино, слот 09:00:

```json
{"station_id": "devyatkino", "ts": "2026-08-31T09:00:00+03:00", "horizon_min": 60,
 "q10": 6910, "q50": 7178, "q90": 7552, "baseline": 8295, "is_anomaly": true,
 "model_version": "lgbm_q_ratio_v1/lgbm_holdout"}
```

- `ts` — начало часового слота, местное время.
- `q10` / `q50` / `q90` — квантили входов на станцию за час.
- `baseline` — норма (медиана 4 последних таких же дней).
- `is_anomaly` — |q50 / baseline − 1| выше p80 группы станций и периода суток (порог выбран на бэктесте,
  [docs/backtest.md](docs/backtest.md), раздел «Флаг аномалии»).
- `model_version` — версия модели и папка в `models/`.

`explanations` — на каждую запись три признака с наибольшим вкладом в q50:

```json
{"station_id": "devyatkino", "ts": "2026-08-31T09:00:00+03:00", "horizon_min": 60,
 "reasons": [{"feature": "r_t", "text": "в последний час поток на 75 % ниже нормы", "effect_pct": -7.1},
             {"feature": "r_line", "text": "вся линия на 25 % ниже нормы", "effect_pct": -4.8},
             {"feature": "b4_tau", "text": "поправка на величину нормы (8 295 входов в 09 ч)", "effect_pct": -2.8}]}
```

`effect_pct` — на сколько процентов признак сдвинул прогноз q50 относительно среднего прогноза модели.

`meta` — какая модель отвечала:

```json
{"model": "lgbm_holdout", "train_start": "2026-02-09", "train_end": "2026-08-24", "out_of_sample": true,
 "warning": null, "now": "2026-08-31 08:59:00", "t0": "2026-08-31T08:00:00+03:00", "config_sha256": "2152d675…"}
```

Время ответа — около 2 секунд: нормы и признаки пересчитываются по данным до `now`.
