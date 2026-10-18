"""Citation parsing: mapping the LLM's inline markers back to real chunks.

The prompt asks the model to cite sources as ``[1]``, ``[2]``, ``[1, 3]``. This
module extracts those markers and resolves them to the chunks that were actually
passed as context, which gives two things the brief asks for:

* a verifiable source list per answer, and
* a signal for *which* retrieved chunks the model leaned on, distinct from the
  ones that were merely retrieved.

It also renumbers. If the model cites only sources 2 and 5 out of 5 supplied, the
user-facing answer is rewritten to cite [1] and [2] against a two-item source
list, so the numbering the user sees is always contiguous and complete.

Stdlib only.
"""

from __future__ import annotations

import re

from app.retrieval.fusion import RetrievedChunk

# Matches [1], [12], [1,2], [1, 2, 3] and the [1][2] adjacency form. Padding
# inside the brackets is tolerated ("[ 1 , 2 ]") because a marker we fail to
# recognise costs the user the whole source list -- the answer gets reported as
# ungrounded even though the model did cite. Still requires a leading digit, so
# "[see above]" and "[]" are correctly left alone.
CITATION_PATTERN = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def extract_citation_numbers(answer: str) -> list[int]:
    """Return cited source numbers in order of first appearance, deduplicated."""
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            num = int(part)
            if num not in seen:
                seen.append(num)
    return seen


def _tidy(text: str) -> str:
    """Clean up whitespace left behind when a marker is removed.

    Stripping "[9]" out of "and [9]." leaves "and ." -- collapse the runs and
    pull punctuation back onto the preceding word.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\( +", "(", text)
    # A parenthetical that held nothing but citation boilerplate -- "(see [9])"
    # becomes "(see)" once the marker is stripped -- should go entirely.
    text = re.sub(
        r"\(\s*(?:see|sources?|refs?|ref\.|cf\.?|from|per)?[\s,;:]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,;:!?)])", r"\1", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def resolve_citations(
    answer: str, context_chunks: list[RetrievedChunk]
) -> tuple[str, list[RetrievedChunk], list[int]]:
    """Resolve inline markers against the supplied context.

    Returns ``(rewritten_answer, cited_chunks, invalid_numbers)`` where
    ``cited_chunks`` is ordered by first mention and ``invalid_numbers`` holds
    out-of-range markers (a hallucinated ``[9]`` when only 5 sources existed).

    If the model cited nothing, the answer is returned untouched and
    ``cited_chunks`` is empty -- the caller decides whether to surface all
    retrieved chunks as unconfirmed context.
    """
    cited_numbers = extract_citation_numbers(answer)
    if not cited_numbers:
        return answer, [], []

    valid: list[int] = []
    invalid: list[int] = []
    for num in cited_numbers:
        if 1 <= num <= len(context_chunks):
            valid.append(num)
        else:
            invalid.append(num)

    if not valid:
        # Every marker was bogus; strip them so we don't show broken references.
        return _tidy(CITATION_PATTERN.sub("", answer)), [], invalid

    # old source number -> new contiguous number, in order of first mention
    renumber = {old: new for new, old in enumerate(valid, start=1)}
    cited_chunks = [context_chunks[old - 1] for old in valid]

    def _replace(match: re.Match[str]) -> str:
        nums = [p.strip() for p in match.group(1).split(",")]
        mapped: list[str] = []
        for raw in nums:
            if raw.isdigit() and int(raw) in renumber:
                token = str(renumber[int(raw)])
                if token not in mapped:
                    mapped.append(token)
        # Drop the marker entirely if none of its numbers were valid.
        return f"[{', '.join(mapped)}]" if mapped else ""

    return _tidy(CITATION_PATTERN.sub(_replace, answer)), cited_chunks, invalid
