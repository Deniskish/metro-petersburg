# Metro Petersburg

Проект готовит внешние признаки для прогнозирования пассажиропотока **1 линии
метро Санкт-Петербурга**: погоду, календарь, городские события и прибытия
железнодорожных поездов к связанным станциям метро.

Единица итоговых данных — **station × timestamp**, например
«Площадь Восстания × 2026-10-08 18:00 Europe/Moscow».
Результат — Pydantic v2 модель `ExternalFeatures`, которую можно сериализовать
в JSON и передать ML-разработчику. Сейчас реализован слой внешних данных;
обучения и запуска ML-модели в репозитории нет.

## Архитектура

```text
Yandex Weather ──→ weather ───────────────────────────┐
YandexGPT ───────→ events ──→ события для станции ─────┤
                             ↑                      │
Yandex Geocoder → locations → координаты и расстояния│
Yandex Rasp ────→ railway ────────────────────────────┤
Локальный календарь → calendar ───────────────────────┘
                            ↓
               build_external_features(...)
                            ↓
                    ExternalFeatures
                            ↓
              ML model — будущий отдельный слой
```

Это схема передачи готовых результатов, а не автоматический сборщик всех API.
Модули вызываются независимо. `features` не делает сетевых запросов: вызывающий
код подбирает данные для станции и времени, затем передаёт их builder.

`locations` дополняет события географическими данными: явно указанный адрес
(предпочтительно) или название площадки → координаты → расстояния Haversine
до 19 станций линии 1. Он возвращает несколько ближайших станций, но не задаёт
радиус влияния и не определяет силу воздействия события. Выбор
`events_for_station` остаётся за вызывающим кодом.

**Passenger flow и lags не входят в ExternalFeatures.** ML-прогноз и расчёт
количества составов метро относятся к следующим слоям системы, которых здесь нет.

## Модули

Подробности контрактов, ограничений и Python API — в README каждого модуля.

| Модуль | Назначение | Вход | Выход | Внешний API |
| --- | --- | --- | --- | --- |
| [external_data/events](external_data/events/README.md) | Извлечение событий из уже полученного текста | Текст, published_at, метаданные источника, LLM provider | `EventExtractionResult` с `list[Event]` | YandexGPT или OpenRouter; есть fake |
| [external_data/weather](external_data/weather/README.md) | Текущая погода и почасовой прогноз | Координаты, горизонт прогноза | `WeatherObservation` или список наблюдений | Yandex Weather |
| [external_data/calendar](external_data/calendar/README.md) | Календарные признаки с учётом источника рабочих/выходных дней | Timestamp и calendar provider / локальная таблица | `CalendarFeatures` | Нет |
| [external_data/locations](external_data/locations/README.md) | Координаты места и ближайшие станции линии 1 | Название, optional address, geocoder, top_k | `LocationResolution` | Yandex Maps Geocoder; есть fake |
| [external_data/railway](external_data/railway/README.md) | Прибытия поездов и электричек в окнах 15/30/60/120 минут | Hub, timestamp, railway provider | `RailwayArrival` и `RailwayFeatures` | Yandex Rasp; есть fake |
| [external_data/features](external_data/features/README.md) | Единый объект для станции и времени | Готовые weather, calendar, events, railway | `ExternalFeatures` | Нет |

## API

Здесь перечислены API, которые используют существующие адаптеры:

- **YandexGPT / Yandex AI Studio** — превращает сырой текст в `Event` через
  OpenAI-compatible endpoint `https://ai.api.cloud.yandex.net/v1`.
  Возвращает structured JSON с Pydantic validation. Адрес разрешён только при
  наличии в исходном тексте; parser проверяет его подстрокой с нормализацией
  регистра и пробелов. `source_fragment` проверяется строго посимвольно.
- **Yandex Weather** — текущая погода и почасовые точки прогноза через
  `https://api.weather.yandex.ru/graphql/query`; авторизация заголовком
  `X-Yandex-Weather-Key`. Интерполяции до 15 минут и запроса исторической погоды нет.
- **Yandex Maps Geocoder** — адрес → координаты через
  `https://geocode-maps.yandex.ru/v1/`. Есть отдельные проверки результатов для
  адреса и названия площадки; сомнительный результат может быть отклонён.
- **Yandex Rasp** — прибытия поездов/электричек через
  `https://api.rasp.yandex-net.ru/v3.0/schedule/`, `event=arrival`.
  Используются train/suburban, pagination и обе даты при переходе окна через полночь.
- **Calendar** — локальный модуль, не API. Для CLI нужна явная таблица дат;
  одна demo-дата в репозитории не заменяет производственный календарь России.

Ключи этих четырёх сервисов **разные**. Подключение каждого сервиса описано в
соответствующем README выше. Для событий также реализован альтернативный
OpenRouter provider; он не обязателен для Yandex smoke tests.

## Установка

Нужен Python **3.11+**. Все команды ниже выполняются из корня репозитория.
Пример для bash/zsh:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install \
  -r external_data/events/requirements.txt \
  -r external_data/weather/requirements.txt \
  -r external_data/calendar/requirements.txt \
  -r external_data/locations/requirements.txt \
  -r external_data/railway/requirements.txt
```

Общего корневого `requirements.txt` нет. `features` использует зависимости
существующих модулей и отдельного requirements не имеет. Основные библиотеки:
Pydantic v2, официальный OpenAI Python SDK и httpx.

Для запуска через pytest установите его отдельно — в requirements модулей
он не перечислен:

```bash
python3 -m pip install pytest
```

Без ключей и интернета после установки доступны demo `calendar`, demo `features`
и все unit tests.

## Environment

Создайте локальный `.env` на основе [.env.example](.env.example), если его ещё нет:

```bash
test -f .env || cp .env.example .env
```

Заполните ключи самостоятельно. Пример конфигурации Yandex без секретов:

```dotenv
YANDEX_API_KEY=
YANDEX_FOLDER_ID=
YANDEX_MODEL=yandexgpt-5-lite
YANDEX_WEATHER_API_KEY=
YANDEX_GEOCODER_API_KEY=
YANDEX_RASP_API_KEY=
```

В текущем файле `.env.example` значение модели — `yandexgpt/latest`; выше показан
вариант локальной настройки с `yandexgpt-5-lite`. Укажите доступное вашему
каталогу короткое имя модели. Provider не выбирает модель по умолчанию:
из `YANDEX_FOLDER_ID` и `YANDEX_MODEL` он собирает
`gpt://<folder_id>/<model>`. Folder ID должен относиться к каталогу с нужными правами.

Шаблон также содержит `OPENROUTER_API_KEY` и `OPENROUTER_MODEL`. Заполняйте их
только для `events --provider openrouter`; для Yandex они не нужны.

Отдельные CLI **не загружают `.env` автоматически**. Перед их запуском
экспортируйте настройки в текущем shell:

```bash
set -a
source .env
set +a
```

`source` выполняет shell-код: используйте только собственный доверенный `.env`.
Общий smoke-скрипт сам читает `.env` как literal assignments, без исполнения
команд и подстановок переменных.

## Запуск модулей

Даты ниже — фиксированные примеры. Для другой публикации передавайте её реальный
`published_at`, для актуального железнодорожного расписания — нужный timestamp.
Datetime должен содержать timezone; основная временная шкала — `Europe/Moscow`.
CLI печатают JSON в stdout, диагностику — в stderr.

### Events: YandexGPT

```bash
python3 -m external_data.events.cli \
  --provider yandex \
  --text "12 октября в 19:30 в БКЗ Октябрьский по адресу Лиговский проспект, 6 состоится концерт Barcelona Flamenco Ballet" \
  --published-at "2026-10-07T12:00:00+03:00" \
  --source-type telegram
```

Также доступны `--source-url` и `--debug`. Одно сообщение может дать несколько
событий. Неизвестные значения остаются null; посещаемость не выводится из
вместимости площадки. Модуль не скачивает посты и не подключается к Telegram.
Для OpenRouter замените provider на `openrouter`; без флага это текущий default.

### Weather: текущая погода и прогноз

```bash
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351 --hours 2
```

`--hours` принимает 1–48. Базовый запрос не включает `feelsLike`; опция
`--with-feels-like` требует соответствующего доступа по тарифу и на Test может
вернуть GraphQL error. У текущей погоды `precipitation=null`; количество осадков
берётся из почасового прогноза. Пропуски не заполняются выдуманными значениями.

### Calendar: локальная demo-таблица

```bash
python3 -m external_data.calendar.cli \
  --timestamp "2026-10-10T18:00:00+03:00" \
  --calendar-file external_data/calendar/examples/demo_calendar.json
```

Для других дат передайте собственную таблицу производственного календаря.
Неизвестная дата вызывает ошибку; рабочие субботы и переносы не угадываются.

### Locations: адрес и ближайшие станции

```bash
python3 -m external_data.locations.cli \
  --location "БКЗ Октябрьский" \
  --address "Лиговский проспект, 6" \
  --top-k 3
```

Адрес имеет приоритет. Без `--address` используется название с проверкой
кандидатов и ограниченными fallback-запросами. `--debug` помогает проверить
фактически найденный адрес и координаты. Возвращаются расстояния в метрах,
не маршруты и не время пути. Кэш координат станций живёт только в памяти процесса.

### Railway: прибытия к вокзалу

```bash
python3 -m external_data.railway.cli \
  --hub "Московский вокзал" \
  --timestamp "2026-10-08T18:00:00+03:00"
```

Доступны четыре hub: Московский вокзал → Площадь Восстания, Финляндский вокзал
→ Площадь Ленина, Балтийский вокзал → Балтийская, Девяткино → Девяткино.
Окна включают границы: `T <= arrival_time <= T + window`.
Это прибытия **железнодорожных поездов и электричек**, не составов метро.

### Features: итоговый объект без API

```bash
python3 -m external_data.features.cli
```

CLI использует только локальные фиксированные demo-значения, включая расписание
и уже отобранные для станции события. Это демонстрация контракта, не сбор
актуальных данных всех сервисов.

## ExternalFeatures

Итог содержит `station`, `timestamp` и четыре блока: `weather`, `calendar`,
`events`, `railway`. `locations` служит отдельным enrichment для событий;
координаты и расстояния не входят в этот ML-контракт.

Builder принимает уже подготовленные объекты:

```python
from external_data.features import build_external_features

# timestamp: aware datetime
# weather: WeatherObservation | None
# calendar: CalendarFeatures для timestamp
# events_for_station: list[Event], отобранные вызывающим кодом
# railway: RailwayFeatures | None для station и timestamp
result = build_external_features(
    timestamp=timestamp,
    station="Площадь Восстания",
    weather=weather,
    calendar=calendar,
    events=events_for_station,
    railway=railway,
)
payload = result.model_dump(mode="json")
```

Важные правила:

- Для реального ML задавайте `station != null`. Optional station сохранён
  для обратной совместимости; с переданным railway станция обязательна.
- `calendar.timestamp` и `railway.timestamp` должны совпадать с моментом
  итогового объекта, а `railway.metro_station` — с `station`.
- Подбор погоды на нужное время и отбор событий по времени/станции выполняются
  до builder. Builder не вычисляет длительность или влияние события.
- `weather=None` даёт null во всех погодных полях; `railway=None` допустим,
  например для станции без учитываемого вокзала. Null не означает ноль.
- `event_count` — длина списка, `event_types` — уникальные ненулевые типы в
  порядке появления. `total_expected_people` суммирует только известные
  значения; если нет ни одного, результат null. Это не оценка пассажиропотока.
- Счётчики railway копируются без пересчёта. Его `minutes_to_next_arrival`
  может быть null или превышать 120, если ближайший загруженный рейс позже окон.

Полный JSON текущего demo CLI:

```json
{
  "timestamp": "2026-10-10T18:00:00+03:00",
  "station": "Площадь Восстания",
  "weather": {
    "temperature": 6.0,
    "feels_like": null,
    "precipitation": 1.2,
    "precipitation_type": "RAIN",
    "wind_speed": 4.3,
    "wind_gust": null,
    "humidity": 85.0,
    "pressure": 750.0,
    "condition": "RAIN"
  },
  "calendar": {
    "day_of_week": 5,
    "hour": 18,
    "minute": 0,
    "is_weekend": true,
    "is_workday": false,
    "is_holiday": false,
    "is_preholiday": null,
    "day_type": "weekend"
  },
  "events": {
    "event_count": 2,
    "event_types": ["concert", "festival"],
    "total_expected_people": 1000,
    "has_event": true
  },
  "railway": {
    "railway_name": "Московский вокзал",
    "arrivals_next_15m": 0,
    "arrivals_next_30m": 1,
    "arrivals_next_60m": 3,
    "arrivals_next_120m": 7,
    "train_arrivals_next_30m": 1,
    "suburban_arrivals_next_30m": 0,
    "minutes_to_next_arrival": 18.0
  }
}
```

## Testing

Все unit tests запускаются из корня, без интернета и реальных API keys:

```bash
pytest
```

Тесты написаны на `unittest` и совместимы с pytest. Варианты запуска:

```bash
python3 -m pytest
python3 -m unittest discover
```

### Общий smoke test

Существующий [scripts/smoke_external_apis.sh](scripts/smoke_external_apis.sh)
выполняет четыре **реальных API-проверки** и две локальные demo-проверки:

```bash
./scripts/smoke_external_apis.sh
```

Скрипт сам переходит в корень и читает `.env`, проверяет шесть Yandex-переменных
с выводом только `SET` / `MISSING`, запускает Events, Weather, Locations,
Railway, Calendar и Features. Для каждого сохраняет stdout, exit code и stderr
при ошибке или наличии диагностики. После FAIL остальные проверки продолжаются.

Результаты:

- `artifacts/external_api_smoke_<YYYY-MM-DD_HHMMSS>.md` — общий отчёт;
- одноимённая директория — очищенные от секретов `.stdout.json`, `.stderr.txt`
  и `.exitcode` для отдельных проверок.

`PASS` означает exit code 0, а не доказательство смысловой точности extraction
или геокодинга. Timeout каждой проверки — 300 секунд; общий exit code ненулевой
при сбое проверки или загрузки конфигурации. Smoke расходует API-квоты и
использует фиксированные даты из скрипта; отдельные CLI выше позволяют выбрать
другие даты без изменения скрипта. Календарь и features в smoke остаются demo.

## Что передаётся Макару / ML

Команда внешних данных передаёт `ExternalFeatures` для станции и момента времени.
ML-разработчик отдельно добавляет историю пассажиропотока, лаги и baseline:

```text
ExternalFeatures + passenger flow + lags + baseline
                          ↓
                  CatBoost / LightGBM
```

Это целевой следующий этап, а не существующий обучающий pipeline.
`external_data` не обучает модель и не возвращает ML prediction.
Преобразование JSON в ML-таблицу, кодирование категорий и обработка пропусков
определяются ML-слоем; `null` нельзя автоматически трактовать как нулевой спрос.

## Что пока НЕ делает проект

- Не считает passenger flow и lag features.
- Не обучает и не запускает CatBoost/LightGBM, не рассчитывает baseline.
- Не рассчитывает количество составов метро.
- Не делает dispatcher recommendation.
- Не запускает scheduler или непрерывный сбор данных.
- Не является FastAPI-сервисом и не хранит данные в БД.
- Не скачивает афиши, не выполняет web scraping и не подключает Telegram API.
- Не определяет радиус/силу влияния события и не строит маршруты до метро.

## Типовые ошибки

| Симптом | Что проверить |
| --- | --- |
| Environment variable missing / MISSING | Нужная переменная заполнена и экспортирована. Файл `.env` сам по себе не загружается отдельными CLI. |
| `.env` заполнен, но CLI не видит ключ | Выполните `set -a; source .env; set +a` в том же shell, где запускаете CLI, либо используйте общий smoke-скрипт. |
| HTTP 401/403 | Ключ относится к нужному сервису, есть доступ и квота. Для YandexGPT проверьте folder ID, права и выбранную модель. |
| Новый ключ ещё не работает | Проверьте статус активации в кабинете соответствующего API; активация может быть не мгновенной. Затем повторите запрос. |
| HTTP 429 / timeout / ошибка сети | Проверьте квоту, соединение и доступность сервиса; повторите позже. Не печатайте URL с apikey для диагностики. |
| Weather GraphQL access error | Проверьте доступ к полю по тарифу; попробуйте без `--with-feels-like`. |
| Место не найдено с достаточной уверенностью | Передайте явно известный адрес через `--address`, проверьте кандидатов с `--debug`; не подставляйте первые попавшиеся координаты. |
| Календарная дата неизвестна | В локальной таблице должна быть запрошенная дата. Demo-файл покрывает только 2026-10-10. |
| Railway не совпадает со station/timestamp | Передавайте признаки именно нужной станции и того же момента времени. |

## Security

- `.env` и файлы `artifacts/` уже исключены через [.gitignore](.gitignore).
  В [.env.example](.env.example) должны оставаться только пустые ключи и примеры настроек.
- Не коммитьте секреты, локальное виртуальное окружение и реальные ключи.
- Не логируйте API keys, Authorization и полные URL с `apikey`.
- Используйте штатный `--debug` и smoke-отчёты с маскированием секретов.
  Перед передачей отчёта проверьте, нет ли в нём чувствительного текста источника.
- Не публикуйте `.env` и не вставляйте его содержимое в задачи или логи.
