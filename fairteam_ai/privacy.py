from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}|\d{2,3}[-\s.]\d{3,4}[-\s.]\d{4})(?!\d)")
STUDENT_ID_RE = re.compile(r"(?<!\d)(?:20\d{2})[-\s.]?\d{4,8}(?!\d)")
URL_TOKEN_RE = re.compile(r"(https?://[^\s]+)(?:token|access_token|key|secret)=([^\s&]+)", re.I)


def redact_text(text: object, *, keep_names: bool = True) -> str:
    """Mask privacy-sensitive strings before showing logs to a wider audience.

    The project should not expose raw emails, phone numbers, student numbers, or
    URL tokens in professor/team reports. Names are intentionally preserved by
    default because member-level accountability is the purpose of the tool.
    """
    s = "" if text is None else str(text)
    s = EMAIL_RE.sub("[EMAIL_REDACTED]", s)
    s = PHONE_RE.sub("[PHONE_REDACTED]", s)
    s = STUDENT_ID_RE.sub("[STUDENT_ID_REDACTED]", s)
    s = URL_TOKEN_RE.sub(r"\1token=[TOKEN_REDACTED]", s)
    if not keep_names:
        # Conservative Korean/English name-like masking for exported public demos.
        s = re.sub(r"(?<!\w)[가-힣]{2,4}(?!\w)", "[NAME]", s)
        s = re.sub(r"(?<!\w)[A-Z][a-z]{2,12}\s[A-Z][a-z]{2,12}(?!\w)", "[NAME]", s)
    return s


def redact_dataframe(df: pd.DataFrame, *, columns: Iterable[str] | None = None, keep_names: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    target_cols = list(columns) if columns is not None else list(out.columns)
    for col in target_cols:
        if col in out.columns and (out[col].dtype == object or pd.api.types.is_string_dtype(out[col])):
            out[col] = out[col].map(lambda x: redact_text(x, keep_names=keep_names))
    return out
