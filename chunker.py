"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


MAX_CHUNK_CHARS = 1000   # sections longer than this get split on paragraphs
MIN_CHUNK_CHARS = 150    # sections shorter than this merge into a neighbour


def _sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(heading, body)]). The intro under '# Title' has heading ''."""
    title = ""
    sections: list[tuple[str, str]] = []
    heading, body = "", []

    def flush():
        joined = "\n".join(body).strip()
        if joined:
            sections.append((heading, joined))

    for line in text.split("\n"):
        if line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        elif line.startswith("# ") and not title:
            title = line[2:].strip()
        else:
            body.append(line)
    flush()
    return title, sections


def _split_long(body: str) -> list[str]:
    """Split an oversized body on paragraph breaks, falling back to sentence ends."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if len(para) <= MAX_CHUNK_CHARS:
            units.append(para)
        else:
            units.extend(re.split(r"(?<=[.!?])\s+", para))

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split each document on its '##' headings, one chunk per section.

    Every chunk is prefixed with "<Document title> - <section heading>" so a
    chunk like "Getting there" still says which city it is about. Sections over
    MAX_CHUNK_CHARS are split on paragraph (then sentence) boundaries, and
    sections under MIN_CHUNK_CHARS are merged into a neighbour. The intro
    paragraph under the '# Title' line is its own chunk.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        title, sections = _sections(doc.text)
        title = title or doc.source.rsplit(".", 1)[0]

        merged: list[list[str]] = []
        pending: list[str] | None = None   # short leading section waiting for a next one
        for heading, body in sections:
            if pending is not None:
                ph, pb = pending
                body = f"{ph}: {pb}\n\n{body}" if ph else f"{pb}\n\n{body}"
                pending = None
            if len(body) < MIN_CHUNK_CHARS:
                if merged:
                    merged[-1][1] += f"\n\n{heading}: {body}" if heading else f"\n\n{body}"
                else:
                    pending = [heading, body]
                continue
            merged.append([heading, body])
        if pending is not None:
            merged.append(pending)

        index = 0
        for heading, body in merged:
            label = f"{title} - {heading}" if heading else title
            pieces = _split_long(body) if len(body) > MAX_CHUNK_CHARS else [body]
            for piece in pieces:
                chunks.append(
                    Chunk(
                        text=f"{label}\n{piece}",
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::split_documents",
                    )
                )
                index += 1
    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
