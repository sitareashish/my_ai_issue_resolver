"""
Context Agent — hybrid semantic + BM25 search with cross-encoder re-ranking.

Pipeline:
  1. Build query from issue title + body + labels + first comment
  2. Dense vector search via Qdrant (top-K)
  3. BM25 keyword search over in-memory index
  4. Fuse scores → top-K candidates
  5. Cross-encoder re-rank → final top-N
  6. Dependency trace: resolve imports for each chunk
  7. Compute confidence score (symbol coverage)
  8. Return ContextPayload
"""
from __future__ import annotations

import re
from pathlib import Path

from src.config import settings
from src.indexer.embedding_pipeline import EmbeddingPipeline
from src.indexer.hybrid_search import HybridSearchIndex, fuse_results
from src.schema import CodeChunk, ContextPayload, GitHubIssue, SystemState
from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _rerank_ok = True
    log.info("cross_encoder_loaded")
except Exception as exc:
    _rerank_ok = False
    log.warning("cross_encoder_unavailable", error=str(exc))

# Stopwords excluded from symbol-coverage calculation
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "should", "when",
    "will", "from", "into", "does", "not", "can", "have", "been",
    "are", "was", "were", "has", "had", "its", "also", "but",
}


class ContextAgent:
    def __init__(self, pipeline: EmbeddingPipeline,
                 bm25_index: HybridSearchIndex | None = None) -> None:
        self._pipeline = pipeline
        self._bm25 = bm25_index

    def run(self, state: SystemState) -> SystemState:
        issue: GitHubIssue = state["issue"]
        repo_path: str = state.get("repo_local_path", "")
        query = self._build_query(issue)

        log.info("context_agent_start", issue=issue.number, confidence_threshold=settings.confidence_threshold)

        # ── Step 1: Dense retrieval ────────────────────────────────────────────
        dense_chunks = self._pipeline.search(query, top_k=settings.context_top_k)

        # ── Step 2: BM25 retrieval ────────────────────────────────────────────
        bm25_results = self._bm25.search(query, top_k=settings.context_top_k) if self._bm25 else []

        # ── Step 3: Hybrid fusion ──────────────────────────────────────────────
        if bm25_results:
            candidates = fuse_results(dense_chunks, bm25_results, top_k=settings.context_top_k)
            strategy = "hybrid"
        else:
            candidates = dense_chunks
            strategy = "dense"

        # ── Step 4: Fallback if too few results ───────────────────────────────
        if len(candidates) < 3:
            log.warning("few_chunks_expanding", found=len(candidates))
            grep_chunks = self._grep_fallback(repo_path, query)
            # Merge without duplicates
            seen = {c.file_path + str(c.start_line) for c in candidates}
            for gc in grep_chunks:
                key = gc.file_path + str(gc.start_line)
                if key not in seen:
                    candidates.append(gc)
                    seen.add(key)
            strategy = "grep_extended"

        # ── Step 5: Cross-encoder re-rank ─────────────────────────────────────
        top_chunks = self._rerank(query, candidates, settings.context_rerank_top_k)

        # ── Step 6: Dependency trace ───────────────────────────────────────────
        if repo_path:
            top_chunks = self._resolve_dependencies(top_chunks, repo_path)

        # ── Step 7: Confidence score ───────────────────────────────────────────
        confidence, found, missing = self._compute_confidence(issue, top_chunks)

        log.info("context_agent_complete",
                 chunks=len(top_chunks), confidence=round(confidence, 3),
                 strategy=strategy, symbols_found=len(found), symbols_missing=len(missing))

        return {
            **state,  # type: ignore[return-value]
            "context": ContextPayload(
                chunks=top_chunks,
                confidence=confidence,
                symbols_found=found,
                symbols_missing=missing,
                search_strategy=strategy,
            ),
        }

    # ── Query construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_query(issue: GitHubIssue) -> str:
        parts = [issue.title, issue.body or ""]
        if issue.labels:
            parts.append("labels: " + ", ".join(issue.labels))
        if issue.comments:
            parts.append(issue.comments[0][:400])
        return "\n".join(p for p in parts if p).strip()[:2_000]

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[CodeChunk], top_k: int) -> list[CodeChunk]:
        if not _rerank_ok or not chunks:
            return sorted(chunks, key=lambda c: c.hybrid_score or c.relevance_score, reverse=True)[:top_k]

        pairs = [(query, c.source[:512]) for c in chunks]
        scores: list[float] = _reranker.predict(pairs).tolist()  # type: ignore[attr-defined]
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_k]:
            chunk.relevance_score = float(score)
            result.append(chunk)
        return result

    # ── Dependency resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_dependencies(chunks: list[CodeChunk], repo_root: str) -> list[CodeChunk]:
        """
        For Python chunks, read the file's top-level imports and prepend them
        so the Developer Agent has full import context.
        """
        enriched: list[CodeChunk] = []
        for chunk in chunks:
            if chunk.language != "python":
                enriched.append(chunk)
                continue
            abs_path = Path(repo_root) / chunk.file_path
            try:
                file_source = abs_path.read_text(encoding="utf-8", errors="replace")
                imports = "\n".join(
                    line for line in file_source.splitlines()
                    if line.startswith(("import ", "from ")) and len(line) < 200
                )
                if imports and imports not in chunk.source:
                    chunk.source = imports + "\n\n" + chunk.source
            except OSError:
                pass
            enriched.append(chunk)
        return enriched

    # ── Confidence scoring ────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        issue: GitHubIssue, chunks: list[CodeChunk]
    ) -> tuple[float, list[str], list[str]]:
        if not chunks:
            return 0.0, [], []

        text = f"{issue.title} {issue.body}"
        identifiers = {
            w for w in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text)
            if w not in _STOPWORDS
        }
        if not identifiers:
            return 0.8, [], []

        combined = "\n".join(c.source for c in chunks)
        found   = [w for w in identifiers if w in combined]
        missing = [w for w in identifiers if w not in combined]
        score   = min(len(found) / len(identifiers), 1.0)
        return score, found, missing

    # ── Grep fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _grep_fallback(repo_path: str, query: str) -> list[CodeChunk]:
        if not repo_path:
            return []
        import subprocess
        keywords = [w for w in re.findall(r"[a-zA-Z_]\w{3,}", query)][:5]
        if not keywords:
            return []
        pattern = "|".join(re.escape(k) for k in keywords)
        try:
            r = subprocess.run(
                ["grep", "-rn", "--include=*.py", "-E", pattern, repo_path],
                capture_output=True, text=True, timeout=10,
            )
            lines = r.stdout.strip().splitlines()[:20]
        except Exception:
            return []

        chunks: list[CodeChunk] = []
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) == 3:
                fpath, lineno, src = parts[0], int(parts[1]), parts[2]
                try:
                    rel = str(Path(fpath).relative_to(repo_path))
                except ValueError:
                    rel = fpath
                chunks.append(CodeChunk(
                    file_path=rel, start_line=lineno, end_line=lineno,
                    language="python", chunk_type="module",
                    source=src.strip(), relevance_score=0.1,
                ))
        return chunks


# ── LangGraph node ─────────────────────────────────────────────────────────────

_pipeline: EmbeddingPipeline | None = None
_bm25_index: HybridSearchIndex | None = None


def context_agent_node(state: SystemState) -> SystemState:
    global _pipeline, _bm25_index
    if _pipeline is None:
        from src.indexer.ast_chunker import RawChunk
        _pipeline = EmbeddingPipeline()
        # Build BM25 index from Qdrant scroll
        try:
            all_chunks_raw = _pipeline.scroll_all()
            # Convert CodeChunk → RawChunk for BM25
            raw_chunks = [
                type("RC", (), {  # duck-type RawChunk
                    "file_path": c.file_path, "start_line": c.start_line,
                    "end_line": c.end_line, "language": c.language,
                    "chunk_type": c.chunk_type, "source": c.source,
                    "chunk_id": f"{c.file_path}:{c.start_line}",
                })()
                for c in all_chunks_raw
            ]
            _bm25_index = HybridSearchIndex(raw_chunks)  # type: ignore[arg-type]
        except Exception as exc:
            log.warning("bm25_index_build_failed", error=str(exc))

    return ContextAgent(_pipeline, _bm25_index).run(state)
