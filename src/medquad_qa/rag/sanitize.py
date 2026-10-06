"""Neutralise prompt-structure look-alikes in untrusted text (retrieved evidence and user questions).

Evidence is data, never instructions. We cannot make a model ignore instructions, but we can stop text
from *forging prompt structure*: evidence delimiters, chat special tokens, citation labels and record IDs.
Medical content (numbers, negations, units, punctuation) is left untouched.
"""

from __future__ import annotations

import re
import unicodedata

SANITIZER_VERSION = "san-v2"

# <evidence ...>, </evidence>, <question>, <system>, ... (any case/whitespace).
_TAG_RE = re.compile(
    r"<\s*/?\s*(evidence|question|system|user|assistant|instructions?|context|document|tool)\b[^<>]{0,200}>",
    re.IGNORECASE,
)
# Chat-template special tokens: <|im_start|>, <|endoftext|>, <\uff5c...\uff5c>, [INST], <<SYS>>, <s>, </s>.
_SPECIAL_RE = re.compile(r"<\s*[|\uff5c][^<>]{0,40}[|\uff5c]\s*>|\[/?INST\]|<<\s*/?SYS\s*>>|</?s>", re.IGNORECASE)
# Citation labels like [E1], [ e 12 ], [E1, E2] and full-width variants.
_LABEL_RE = re.compile(r"[\[\uff3b\u3010]\s*(E\s*\d+(?:\s*,\s*E\s*\d+)*)\s*[\]\uff3d\u3011]", re.IGNORECASE)
# Record-ID look-alikes: mq-<hex>.
_RID_RE = re.compile(r"\bmq-(?=[0-9a-f]{4,})", re.IGNORECASE)
_SENTINEL_RE = re.compile(r"INSUFFICIENT[\s_]*EVIDENCE", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Fullwidth/CJK look-alikes of structural punctuation, folded to ASCII before matching (F-004).
_FOLD = str.maketrans(
    {
        "\uff1c": "<",
        "\uff1e": ">",
        "\uff3b": "[",
        "\uff3d": "]",
        "\u3010": "[",
        "\u3011": "]",
        "\uff5c": "|",
        "\u2329": "<",
        "\u232a": ">",
        "\u3008": "<",
        "\u3009": ">",
        "\ufe64": "<",
        "\ufe65": ">",
        "\uff25": "E",
        "\uff45": "e",
    }
)


def _strip_format_chars(text: str) -> str:
    """Remove invisible format characters (Unicode category Cf: zero-width, bidi controls, BOM, word joiner)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def neutralize(text: str) -> str:
    """Return ``text`` with structural look-alikes defanged. Idempotent."""
    t = unicodedata.normalize("NFC", text)
    t = _strip_format_chars(t)
    t = _CONTROL_RE.sub(" ", t)
    t = t.translate(_FOLD)
    t = _SPECIAL_RE.sub(
        lambda m: m.group(0).replace("<", "\u2039").replace(">", "\u203a").replace("[", "(").replace("]", ")"), t
    )
    t = _TAG_RE.sub(lambda m: m.group(0).replace("<", "\u2039").replace(">", "\u203a"), t)
    t = _LABEL_RE.sub(lambda m: "(" + re.sub(r"\s+", "", m.group(1)) + ")", t)
    t = _RID_RE.sub("mq\u2010", t)  # U+2010 HYPHEN: no longer parses as a record ID
    t = _SENTINEL_RE.sub("insufficient evidence", t)
    return t
