"""Building safe FTS5 MATCH queries from free-text questions.

FTS5 has its own query grammar: bare user input containing quotes, ``AND``,
``NEAR``, ``*``, ``^`` or a stray ``-`` is either a syntax error or silently
means something the user never asked for. Since questions come straight from a
text box, the query must be constructed rather than interpolated.

Approach: reduce the question to word tokens, drop stopwords, wrap each token in
double quotes (making it a literal, so no token can be an operator), and join
with ``OR``. Pure stdlib, so it is directly testable.
"""

from __future__ import annotations

import re

# Deliberately short. An aggressive stopword list hurts recall on real questions
# ("What is the *state* of..."), and BM25 already discounts common terms via IDF.
# These are the words that appear in nearly every question and carry no signal.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those
    is are was were be been being am
    do does did doing done
    of in on at to for from by with about into over under
    and or but if then than so because as
    i you he she it we they me my your his her its our their
    what which who whom whose when where why how
    can could will would shall should may might must
    there here s t
    """.split()
)

# `\w+` rather than an explicit [0-9A-Za-z_] class: on str patterns `\w` is
# unicode-aware, so "café" and "naïve" stay whole. An ASCII-only class would cut
# them at the accent ("caf", "na", "ve") and silently fail to match the FTS index,
# which uses SQLite's unicode61 tokenizer and does keep them whole.
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Guardrail: a pasted paragraph would otherwise build a huge OR query that
# matches almost every chunk and drowns the BM25 signal.
MAX_TERMS = 24


def tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens, preserving order and duplicates."""
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text or "")]


def build_fts_match_query(question: str, max_terms: int = MAX_TERMS) -> str | None:
    """Build an FTS5 MATCH expression, or None if nothing searchable remains.

    Returning None is meaningful: the caller should skip keyword search entirely
    rather than run a query that matches everything.
    """
    tokens = tokenize(question)
    if not tokens:
        return None

    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        # Single characters are noise once quoted; FTS5 tokenizes them away anyway.
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break

    if not terms:
        # Question was entirely stopwords ("what is it?"). Fall back to the raw
        # tokens so keyword search still contributes something rather than
        # nothing -- dedup preserved, still length-capped.
        for token in tokens:
            if token not in seen:
                seen.add(token)
                terms.append(token)
            if len(terms) >= max_terms:
                break

    if not terms:
        return None

    # Quoting makes every term a literal string, so "AND"/"NEAR"/"*" from user
    # input can never be parsed as an FTS5 operator.
    return " OR ".join(f'"{term}"' for term in terms)
