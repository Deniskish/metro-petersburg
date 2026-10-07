"""Validated extraction contract; all timestamps carry an explicit timezone."""

from datetime import datetime
from typing import Annotated, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

MOSCOW = ZoneInfo("Europe/Moscow")


def normalize_timestamp(value: object) -> datetime:
    """Reject ambiguous dates/epochs and normalize aware timestamps to Moscow."""
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            raise ValueError("Timestamp must include a date, time and UTC offset")
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be an ISO 8601 string or datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must have an explicit timezone")
    return value.astimezone(MOSCOW)


Timestamp = Annotated[datetime, BeforeValidator(normalize_timestamp)]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_name: str = Field(min_length=1, strict=True)
    event_type: str | None = None
    start_time: Timestamp | None = None
    end_time: Timestamp | None = None
    location_name: str | None = None
    expected_people: int | None = Field(default=None, ge=0, strict=True)
    source_type: str | None = None
    source_url: str | None = None
    published_at: Timestamp | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, strict=True)
    source_fragment: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def chronological_order(self) -> Self:
        if self.start_time and self.end_time and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        return self


class EventExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_event: bool = Field(strict=True)
    events: list[Event]

    @model_validator(mode="after")
    def consistent_event_flag(self) -> Self:
        if self.is_event != bool(self.events):
            raise ValueError("is_event must equal bool(events)")
        return self
