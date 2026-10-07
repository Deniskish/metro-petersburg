"""Общие константы проекта: пути, период, часовой пояс, источники данных."""
from pathlib import Path

import pandas as pd

# --- Пути -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
PREDICTIONS = DATA / "predictions"
REFERENCE = DATA / "reference"            # ручные справочники, в git
EVENTS_MANUAL = REFERENCE / "events_manual.csv"
FIGURES = ROOT / "reports" / "figures"

# --- Период и время ---------------------------------------------------------
# Все стыковки — по ts_utc; локальное время — только для календаря (раздел 3 ТЗ).
TZ_LOCAL = "America/New_York"
START_LOCAL = pd.Timestamp("2023-01-01 00:00")  # включительно, местное время
END_LOCAL = pd.Timestamp("2024-12-31 23:00")    # включительно, местное время
if END_LOCAL.year > 2024:
    raise ValueError("Данные 2025 года отложены для финального теста — не качаем их на этапе 1")

START_UTC = START_LOCAL.tz_localize(TZ_LOCAL).tz_convert("UTC")  # 2023-01-01 05:00Z
END_UTC = END_LOCAL.tz_localize(TZ_LOCAL).tz_convert("UTC")      # 2025-01-01 04:00Z

# --- MTA Subway Hourly Ridership 2020–2024 ---------------------------------
MTA_URL = "https://data.ny.gov/resource/wujg-7c2s.json"
LINE7_REGEX = r"[(,]\s*7\s*[,)]"
MTA_COMPLEX_PROBE = ("2024-06-03T00:00:00", "2024-06-09T23:00:00")  # неделя для списка комплексов

# Комплексы линии 7 в порядке по линии: Flushing-Main St → 34 St-Hudson Yards.
# Список сверен со схемой; download проверяет, что регулярка находит ровно его.
LINE7_COMPLEXES = [
    ("447", "Flushing-Main St (7)"),
    ("448", "Mets-Willets Point (7)"),
    ("449", "111 St (7)"),
    ("450", "103 St-Corona Plaza (7)"),
    ("451", "Junction Blvd (7)"),
    ("452", "90 St-Elmhurst Av (7)"),
    ("453", "82 St-Jackson Hts (7)"),
    ("616", "74-Broadway (7)/Jackson Hts-Roosevelt Av (E,F,M,R)"),
    ("455", "69 St (7)"),
    ("456", "61 St-Woodside (7)"),
    ("457", "52 St (7)"),
    ("458", "46 St-Bliss St (7)"),
    ("459", "40 St-Lowery St (7)"),
    ("460", "33 St-Rawson St (7)"),
    ("461", "Queensboro Plaza (7,N,W)"),
    ("606", "Court Sq (E,G,M,7)"),
    ("463", "Hunters Point Av (7)"),
    ("464", "Vernon Blvd-Jackson Av (7)"),
    ("610", "Grand Central-42 St (S,4,5,6,7)"),
    ("609", "Bryant Pk (B,D,F,M)/5 Av (7)"),
    ("611", "Times Sq-42 St (N,Q,R,W,S,1,2,3,7)/42 St (A,C,E)"),
    ("471", "34 St-Hudson Yards (7)"),
]
LINE7_IDS = [cid for cid, _ in LINE7_COMPLEXES]

# Пересадочные комплексы: входы включают пассажиров других линий.
TRANSFER_COMPLEX_IDS = {"616", "461", "606", "610", "609", "611"}

# --- Open-Meteo -------------------------------------------------------------
QUEENS = (40.75, -73.88)
WEATHER_VARS = ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "weather_code"]
WEATHER_URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "hist_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}
# место → (координаты, start_date, end_date в GMT); точки СПб заданы ниже, в блоке СПб
WEATHER_PLACES = {"queens": (QUEENS, f"{START_UTC:%Y-%m-%d}", f"{END_UTC:%Y-%m-%d}")}

# --- MLB Stats API ----------------------------------------------------------
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
METS_TEAM_ID = 121
CITI_FIELD_VENUE_ID = 3289
MLB_GAME_TYPES = ["R", "F", "D", "L", "W"]  # регулярка и плей-офф; весенние сборы (S) — во Флориде
MLB_SEASONS = [2023, 2024]
GAME_EST_DURATION = pd.Timedelta(hours=3)   # оценка окончания, известная заранее

# --- Праздники --------------------------------------------------------------
HOLIDAY_YEARS = [2023, 2024]

# --- 1 линия СПб: данные организаторов (разделы 1–3 ТЗ) ----------------------
SPB_TZ = "Europe/Moscow"                  # перехода на летнее время нет
SPB_RAW = RAW / "spb"                     # файлы организаторов как есть, в подпапках drive-download-*
SPB_FLOW_FILE = "Пассажиропоток 2026 Линия 1.xlsx"
SPB_MO1_GLOB = "MO1*.xls"                 # HTML в KOI8-R с расширением .xls

# Дата — строка «ДД.ММ.ГГГГ ЧЧ», час — начало интервала; автоматический разбор запрещён.
# Шаг и формат — параметры: при 10-минутной выгрузке из АСКОП М меняются вместе.
SPB_TS_FORMAT = "%d.%m.%Y %H"
SPB_FREQ = "h"
SPB_START_LOCAL = pd.Timestamp("2026-01-01 00:00")  # включительно, местное время
SPB_END_LOCAL = pd.Timestamp("2026-09-30 00:00")    # включительно, местное время
SPB_START_UTC = SPB_START_LOCAL.tz_localize(SPB_TZ).tz_convert("UTC")  # 2025-12-31 21:00Z
SPB_END_UTC = SPB_END_LOCAL.tz_localize(SPB_TZ).tz_convert("UTC")      # 2026-09-29 21:00Z

SPB_CLOSED_HOURS = (1, 2, 3, 4)           # метро закрыто (местное время)
SPB_SERVICE_DAY_START = 5                 # сутки метро: 05:00 … 00:59, час 0 — конец суток
# Разовое закрытие вестибюля: вход ≤ MAX_ENTRIES при медиане этого вестибюля и часа ≥ MIN_NORM
SPB_CLOSURE_MAX_ENTRIES = 2
SPB_CLOSURE_MIN_NORM = 50

# Справочники (в git); slug-id станций — согласовать с человеком 3 (симулятор)
SPB_STATIONS_CSV = REFERENCE / "spb_line1_stations.csv"
SPB_VESTIBULES_CSV = REFERENCE / "spb_line1_vestibules.csv"
SPB_INCIDENTS_CSV = REFERENCE / "incidents.csv"
SPB_EVENTS_CSV = REFERENCE / "events_spb.csv"
LINE1_OPERATIONS_YAML = REFERENCE / "line1_operations.yaml"

SPB_HOURLY = INTERIM / "spb_line1_hourly.parquet"
MO1_OUT = INTERIM / "mo1_reports.parquet"

# Погода СПб: даты в GMT — 00–02 MSK 1 января приходятся на 31 декабря UTC
SPB_COORDS = (59.94, 30.31)
WEATHER_PLACES["spb"] = (SPB_COORDS, f"{SPB_START_UTC:%Y-%m-%d}", f"{SPB_END_LOCAL:%Y-%m-%d}")  # 2025-12-31 … 2026-09-30

# Производственный календарь РФ с переносами: 0 — рабочий, 1 — нерабочий, 2 — сокращённый (pre=1)
ISDAYOFF_URL = "https://isdayoff.ru/api/getdata"
CALENDAR_YEAR = 2026
CALENDAR_OUT = INTERIM / f"calendar_ru_{CALENDAR_YEAR}.parquet"
