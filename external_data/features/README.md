# Единый контракт внешних признаков

`external_data.features` объединяет готовые результаты `weather`, `calendar`
и `events` в Pydantic v2 модель `ExternalFeatures`, пригодную для передачи в ML.
Модуль сам не получает данные и не вызывает API. Используются зависимости уже
установленных модулей; дополнительные пакеты не нужны.

## Использование

```python
from external_data.features import build_external_features

# timestamp: datetime с часовым поясом
# weather: WeatherObservation | None
# calendar: CalendarFeatures для того же момента времени
# events: list[Event]
result = build_external_features(
    timestamp=timestamp,
    weather=weather,
    calendar=calendar,
    events=events,
    station=None,
)
print(result.model_dump_json(indent=2))
```

Входной `timestamp` обязан содержать часовой пояс: naive datetime отклоняется.
Время переводится в `Europe/Moscow` без изменения момента времени.
`calendar` обязателен, его timestamp должен совпадать с целевым моментом;
несовпадение вызывает `ValueError`. Календарные значения копируются без
пересчёта, включая рабочие субботы и переносы из исходного календаря.
`day_of_week`: понедельник — 0, воскресенье — 6.

Погодные значения и их единицы сохраняются из `WeatherObservation`;
timestamp и координаты наблюдения не входят во вложенный объект `weather`.
При `weather=None` все девять погодных полей равны `null`.
Подбор наблюдения или прогноза для нужного времени выполняет вызывающий код.

События агрегируются по всему переданному списку:

- `event_count` — длина списка, `has_event` — есть ли хотя бы одно событие.
- `event_types` — уникальные ненулевые типы в порядке первого появления.
- `total_expected_people` — сумма только известных значений. Если известных
  значений нет (включая пустой список), результат `null`. Известный ноль
  сохраняется как `0`. При частично неизвестной посещаемости сумма неполная.

Вызывающий код отвечает за выбор актуальных событий и удаление дубликатов.
`station` — необязательная метка, а не результат привязки событий к станции.
Входные объекты не изменяются.

## Локальный пример

Из корня репозитория:

```bash
python3 -m external_data.features.cli
```

CLI использует только фиксированные демонстрационные данные, без ключей и
сетевых запросов. Погода и календарь в нём не являются реальными наблюдениями
или проверенным производственным календарём.

```json
{
  "timestamp": "2026-10-10T18:00:00+03:00",
  "station": null,
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
  }
}
```

## Тесты без интернета

```bash
python3 -m unittest discover -s external_data/features/tests
python3 -m unittest discover
```
