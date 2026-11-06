import re


EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d(?:[\s-]?\d){8}(?!\w)")


def redact_pii(text):
    value = text or ""
    value, emails = EMAIL_PATTERN.subn("[EMAIL]", value)
    value, phones = PHONE_PATTERN.subn("[PHONE]", value)
    return value, {"emails": emails, "phones": phones}
