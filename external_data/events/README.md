# Извлечение городских событий

Автономный модуль принимает уже полученный текст и извлекает события через
заменяемый LLM provider. Сеть, crawler, Telegram API, ML и хранилище не входят
в модуль. Python 3.11+, Pydantic v2; тесты используют стандартный `unittest`.

Из корня репозитория:

```sh
python3 -m pip install -r external_data/events/requirements.txt
python3 -m unittest discover -s external_data/events/tests -v
```

## Использование без API

```python
from datetime import datetime
from external_data.events import parse_events
from external_data.events.providers import FakeLLMProvider

text = "Завтра в 19:00 в Ледовом дворце состоится концерт."
provider = FakeLLMProvider({
    "is_event": True,
    "events": [{
        "event_name": "Концерт",
        "event_type": "concert",
        "start_time": "2026-10-08T19:00:00+03:00",
        "location_name": "Ледовый дворец",
        "confidence": 0.95,
        "source_fragment": text,
    }],
})
result = parse_events(
    text,
    published_at=datetime.fromisoformat("2026-10-07T11:30:00+03:00"),
    source_type="telegram",
    source_url="https://t.me/example/42",
    provider=provider,
)
print(result.model_dump_json(indent=2))
```

Результат:

```json
{
  "is_event": true,
  "events": [{
    "event_name": "Концерт",
    "event_type": "concert",
    "start_time": "2026-10-08T19:00:00+03:00",
    "end_time": null,
    "location_name": "Ледовый дворец",
    "expected_people": null,
    "source_type": "telegram",
    "source_url": "https://t.me/example/42",
    "published_at": "2026-10-07T11:30:00+03:00",
    "confidence": 0.95,
    "source_fragment": "Завтра в 19:00 в Ледовом дворце состоится концерт."
  }]
}
```

Для повторных вызовов: `parser = EventParser(provider)`, затем
`parser.parse_events(text, published_at=...)`. Функция `parse_events` требует
явного keyword-аргумента `provider`: скрытого сетевого вызова или mock по
умолчанию нет. FakeLLMProvider всегда возвращает копию заданного словаря и
**не анализирует текст**.

## Контракт провайдера и валидация

Реализуйте `LLMProvider.generate_structured(system_prompt=..., user_prompt=...,
json_schema=...) -> dict` для YandexGPT, OpenRouter или другого API. Парсер
передает `EventExtractionResult.model_json_schema()`. Адаптер может передать
схему в native structured output; адаптация к ограничениям схем конкретного
API остается внутри адаптера. Если API возвращает строку, декодируйте весь
ответ через `json.loads`. Не ищите фигурные скобки и не исправляйте JSON
эвристически. SDK и ключи для локальных тестов не нужны.

Pydantic проверяет структуру, неизвестные поля, неотрицательное целое число
посетителей, confidence в [0, 1], согласованность is_event/events и порядок
начала/окончания. Timestamps должны содержать дату, время и явный часовой пояс;
даты без времени, naive datetime, числовые epochs и невозможные даты отклоняются.
Валидные timestamps переводятся в Europe/Moscow. Парсер требует дословный
source_fragment и устанавливает source_type/source_url/published_at из
аргументов, включая None; LLM не является источником этих метаданных.

Ошибки провайдера, TypeError/ValueError и Pydantic ValidationError передаются
вызывающему коду. Некорректный ответ не превращается в «событий нет».

## Временная и смысловая политика

- Относительные даты вычисляет LLM относительно московского published_at.
  Без него относительная дата неизвестна. Системные часы не используются.
- Дата без года использует год публикации при отсутствии других уточнений.
  «В пятницу» — ближайшая пятница, включая день публикации; контекст прошедшего
  времени имеет приоритет.
- Дата без точного времени, «вечером» и время без даты дают null. Схема не
  хранит частичные даты; подстановка полуночи создала бы ложную точность.
- Перенос дает только новую дату; неизвестная новая дата — null.
- Отмененные мероприятия исключаются. Схема без статуса отмены не поддерживает
  удаление ранее извлеченных событий: для этого downstream потребуется
  отдельный контракт обновлений. Прошедшие мероприятия допускаются.
- Вместимость площадки и стоимость билетов не определяют expected_people.
  Ближайшая станция определяется отдельным компонентом.

В `examples/sample_events.json` — 17 примеров с фиксированными ожидаемыми
ответами. Тесты дат проверяют контракт, передачу контекста и сохранение
ожидаемых значений mock, **а не способность реальной LLM вычислять даты**.
Pydantic и проверка цитаты не доказывают смысловую достоверность извлечения.
Перед использованием реального провайдера эти же примеры следует прогнать
через него и отдельно оценить ошибки дат, переносов, отмен и выдуманных фактов.
