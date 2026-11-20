
from __future__ import annotations

from datetime import datetime

NOLEAP_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
SECONDS_PER_DAY = 86_400


def validate_noleap_datetime(value: datetime, *, name: str = "date-time") -> None:

    if value.tzinfo is not None:
        raise ValueError(f"{name} must not carry a timezone")
    if value.microsecond:
        raise ValueError(f"{name} must have whole-second precision")
    if value.month == 2 and value.day == 29:
        raise ValueError(f"{name} cannot be February 29 in a NO_LEAP calendar")


def noleap_day_ordinal(value: datetime) -> int:

    validate_noleap_datetime(value)
    return (value.year - 1) * 365 + sum(NOLEAP_MONTH_LENGTHS[: value.month - 1]) + value.day - 1


def noleap_absolute_seconds(value: datetime) -> int:

    return noleap_day_ordinal(value) * SECONDS_PER_DAY + (
        value.hour * 3600 + value.minute * 60 + value.second
    )


def noleap_seconds_between(start: datetime, stop: datetime) -> int:

    return noleap_absolute_seconds(stop) - noleap_absolute_seconds(start)


def noleap_datetime_from_seconds(seconds: int) -> datetime:

    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
        raise ValueError("NO_LEAP absolute seconds must be a nonnegative integer")
    ordinal, time_of_day = divmod(seconds, SECONDS_PER_DAY)
    zero_based_year, day_of_year = divmod(ordinal, 365)
    year = zero_based_year + 1
    month = 1
    while day_of_year >= NOLEAP_MONTH_LENGTHS[month - 1]:
        day_of_year -= NOLEAP_MONTH_LENGTHS[month - 1]
        month += 1
    hour, remainder = divmod(time_of_day, 3600)
    minute, second = divmod(remainder, 60)
    return datetime(year, month, day_of_year + 1, hour, minute, second)


def add_noleap_seconds(value: datetime, seconds: int) -> datetime:

    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise TypeError("NO_LEAP increment must be an integer number of seconds")
    result = noleap_absolute_seconds(value) + seconds
    if result < 0:
        raise ValueError("NO_LEAP increment precedes year 1")
    return noleap_datetime_from_seconds(result)


def noleap_calendar_day(value: datetime) -> float:

    validate_noleap_datetime(value)
    day = sum(NOLEAP_MONTH_LENGTHS[: value.month - 1]) + value.day
    seconds = value.hour * 3600 + value.minute * 60 + value.second
    return day + seconds / SECONDS_PER_DAY


def integer_date_and_time(value: datetime) -> tuple[int, int]:

    validate_noleap_datetime(value)
    return (
        value.year * 10_000 + value.month * 100 + value.day,
        value.hour * 3600 + value.minute * 60 + value.second,
    )


__all__ = [
    "NOLEAP_MONTH_LENGTHS",
    "SECONDS_PER_DAY",
    "add_noleap_seconds",
    "integer_date_and_time",
    "noleap_absolute_seconds",
    "noleap_calendar_day",
    "noleap_datetime_from_seconds",
    "noleap_day_ordinal",
    "noleap_seconds_between",
    "validate_noleap_datetime",
]
