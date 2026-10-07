"""Provider-independent orchestration and validation, without network I/O."""

from datetime import datetime

from .models import EventExtractionResult, normalize_timestamp
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .providers.base import LLMProvider


class EventParser:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def parse_events(
        self,
        text: str,
        published_at: datetime | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
    ) -> EventExtractionResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if published_at is not None:
            if not isinstance(published_at, datetime):
                raise TypeError("published_at must be a datetime or None")
            published_at = normalize_timestamp(published_at)
        for name, value in (("source_type", source_type), ("source_url", source_url)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")

        response = self.provider.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(text, published_at, source_type, source_url),
            json_schema=EventExtractionResult.model_json_schema(),
        )
        if not isinstance(response, dict):
            raise TypeError("Provider must return a decoded JSON object")
        result = EventExtractionResult.model_validate(response)
        for event in result.events:
            if event.source_fragment is None or event.source_fragment not in text:
                raise ValueError("source_fragment must be a verbatim source substring")
            # Source metadata is authoritative; never trust model-generated values.
            event.source_type = source_type
            event.source_url = source_url
            event.published_at = published_at
        return result


def parse_events(
    text: str,
    published_at: datetime | None = None,
    source_type: str | None = None,
    source_url: str | None = None,
    *,
    provider: LLMProvider,
) -> EventExtractionResult:
    """Extract with an explicitly injected provider; no implicit API or fake data."""
    return EventParser(provider).parse_events(text, published_at, source_type, source_url)
