"""
Hybrid search: combines BM25 keyword scores with dense vector cosine scores.
Weighted fusion: hybrid = bm25_weight * bm25 + (1 - bm25_weight) * dense

This significantly improves recall for exact identifier matches (e.g. function
names from the issue description) while preserving semantic coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import settings
from src.indexer.ast_chunker import RawChunk
from src.schema import CodeChunk
from src.utils.logger import get_logger

log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class HybridSearchIndex:
    """
    In-memory BM25 index over the current codebase chunks.
    Built once after indexing; queried during context retrieval.
    """

    def __init__(self, chunks: list[RawChunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(c.source) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        log.info("bm25_index_built", documents=len(chunks))

    def search(self, query: str, top_k: int = 15) -> list[tuple[RawChunk, float]]:
        """Return (chunk, bm25_score) sorted by score descending."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores: list[float] = self._bm25.get_scores(tokens).tolist()
        ranked = sorted(
            zip(scores, self._chunks), key=lambda x: x[0], reverse=True
        )
        return [(chunk, score) for score, chunk in ranked[:top_k]]


def fuse_results(
    dense_chunks: list[CodeChunk],
    bm25_results: list[tuple[RawChunk, float]],
    top_k: int = 15,
    bm25_weight: float | None = None,
) -> list[CodeChunk]:
    """
    Combine dense and BM25 results using weighted score fusion.
    Normalises each score set to [0, 1] before combining.
    """
    w_bm25 = bm25_weight if bm25_weight is not None else settings.bm25_weight
    w_dense = 1.0 - w_bm25

    # Normalise dense scores
    dense_scores = {c.file_path + str(c.start_line): c.relevance_score for c in dense_chunks}
    max_dense = max(dense_scores.values(), default=1.0) or 1.0

    # Normalise BM25 scores
    bm25_scores_raw = {
        chunk.file_path + str(chunk.start_line): score for chunk, score in bm25_results
    }
    max_bm25 = max(bm25_scores_raw.values(), default=1.0) or 1.0

    # Build a unified set of candidates
    all_keys: set[str] = set(dense_scores) | set(bm25_scores_raw)

    # Map key → CodeChunk
    chunk_map: dict[str, CodeChunk] = {
        c.file_path + str(c.start_line): c for c in dense_chunks
    }
    for raw_chunk, _ in bm25_results:
        key = raw_chunk.file_path + str(raw_chunk.start_line)
        if key not in chunk_map:
            chunk_map[key] = CodeChunk(
                file_path=raw_chunk.file_path,
                start_line=raw_chunk.start_line,
                end_line=raw_chunk.end_line,
                language=raw_chunk.language,
                chunk_type=raw_chunk.chunk_type,
                source=raw_chunk.source,
            )

    fused: list[tuple[float, CodeChunk]] = []
    for key in all_keys:
        d = dense_scores.get(key, 0.0) / max_dense
        b = bm25_scores_raw.get(key, 0.0) / max_bm25
        hybrid = w_dense * d + w_bm25 * b
        chunk = chunk_map[key]
        chunk.hybrid_score = hybrid
        chunk.bm25_score = bm25_scores_raw.get(key, 0.0)
        fused.append((hybrid, chunk))

    fused.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in fused[:top_k]]
    log.debug("hybrid_search_complete",
              dense_hits=len(dense_chunks), bm25_hits=len(bm25_results),
              fused_top_k=len(result))
    return result
