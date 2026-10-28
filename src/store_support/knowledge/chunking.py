"""Markdown -> chunks. One chunk per `##` section (policies are short; sections are the
natural citation unit). Pure function."""

from __future__ import annotations

import re

from store_support.knowledge.store import Chunk

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def chunk_markdown(text: str, *, source: str) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else source
    parts = _H2.split(text)
    if len(parts) == 1:
        body = parts[0].strip()
        return [Chunk(source=source, section=title, content=f"{title}\n{body}")] if body else []
    chunks: list[Chunk] = []
    preamble = parts[0].strip()
    if preamble and preamble != f"# {title}":
        chunks.append(Chunk(source=source, section=title, content=preamble))
    for i in range(1, len(parts), 2):
        section, body = parts[i].strip(), parts[i + 1].strip()
        if body:
            chunks.append(Chunk(source=source, section=section, content=f"{title} - {section}\n{body}"))
    return chunks
