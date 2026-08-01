from enum import Enum


class Action(str, Enum):
    EXPLAIN = "explain"
    SUMMARY = "summary"
    RETENTION = "retention"
    EMAIL = "email"
    MARKETING = "marketing"
    FREE_CHAT = "chat"