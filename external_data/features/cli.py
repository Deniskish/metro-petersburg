"""Offline demonstration: python3 -m external_data.features.cli."""

from datetime import datetime

from external_data.calendar.models import CalendarFeatures
from external_data.events.models import Event
from external_data.weather.models import WeatherObservation

from .builder import build_external_features


def main() -> int:
    timestamp = datetime.fromisoformat("2026-10-10T18:00:00+03:00")
    # Fixed demo values; no claims about actual weather or official calendar.
    weather = WeatherObservation(
        timestamp=timestamp, latitude=59.9343, longitude=30.3351,
        temperature=6.0, precipitation=1.2, precipitation_type="RAIN",
        wind_speed=4.3, humidity=85, pressure=750, condition="RAIN",
    )
    calendar = CalendarFeatures(
        timestamp=timestamp, day_of_week=5, hour=18, minute=0,
        is_weekend=True, is_workday=False, is_holiday=False,
        is_preholiday=None, day_type="weekend",
    )
    events = [
        Event(event_name="Демо-концерт", event_type="concert", expected_people=1000),
        Event(event_name="Демо-фестиваль", event_type="festival", expected_people=None),
    ]
    result = build_external_features(timestamp, weather, calendar, events)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
