from enum import IntEnum


class Outcome(IntEnum):
    """Defines the classification outcome."""
    POSITIVE = 1
    NEGATIVE = 0


class Group(IntEnum):
    """Defines the groups involved."""
    UNPRIVILEGED = 0
    PRIVILEGED = 1
