"""Embedding pipeline: OpenAI text-embedding-3-large → Qdrant upsert + search."""
from __future__ import annotations

import asyncio
from typing import Sequence

from openai import OpenAI, AsyncOpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from src.config import settings
from src.indexer.ast_chunker import RawChunk
from src.schema import CodeChunk
from src.utils.logger import get_logger
from src.utils.rate_limiter import call_with_backoff

log = get_logger(__name__)

_BATCH = 64
_MAX_SOURCE_CHARS = 8_000


class EmbeddingPipeline:
    def __init__(self) -> None:
        self._oai  = OpenAI(api_key=settings.openai_api_key)
        self._aoai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._col = settings.qdrant_collection
        self._dim = settings.embedding_dimensions
        self._ensure_collection()

    # ── Collection ────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._col not in existing:
            self._qdrant.create_collection(
                collection_name=self._col,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE, on_disk=True),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10_000),
            )
            log.info("qdrant_collection_created", collection=self._col, dim=self._dim)

    def drop_collection(self) -> None:
        self._qdrant.delete_collection(self._col)
        self._ensure_collection()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_chunks(self, chunks: list[RawChunk]) -> None:
        log.info("indexing_start", chunks=len(chunks))
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            vecs = self._embed([c.source[:_MAX_SOURCE_CHARS] for c in batch])
            points = [
                models.PointStruct(
                    id=abs(hash(c.chunk_id)) % (2 ** 63),
                    vector=v,
                    payload={
                        "chunk_id":   c.chunk_id,
                        "file_path":  c.file_path,
                        "start_line": c.start_line,
                        "end_line":   c.end_line,
                        "language":   c.language,
                        "chunk_type": c.chunk_type,
                        "source":     c.source,
                    },
                )
                for c, v in zip(batch, vecs)
            ]
            self._qdrant.upsert(collection_name=self._col, points=points)
        log.info("indexing_complete", collection=self._col, total=self.count())

    async def index_chunks_async(self, chunks: list[RawChunk]) -> None:
        """Async version using asyncio.gather for parallel embedding batches."""
        tasks = [
            self._embed_async([c.source[:_MAX_SOURCE_CHARS] for c in chunks[i:i+_BATCH]])
            for i in range(0, len(chunks), _BATCH)
        ]
        all_vecs = await asyncio.gather(*tasks)
        for i, (batch_vecs, batch_idx) in enumerate(
            zip(all_vecs, range(0, len(chunks), _BATCH))
        ):
            batch = chunks[batch_idx : batch_idx + _BATCH]
            points = [
                models.PointStruct(
                    id=abs(hash(c.chunk_id)) % (2 ** 63),
                    vector=v,
                    payload={"file_path": c.file_path, "start_line": c.start_line,
                             "end_line": c.end_line, "language": c.language,
                             "chunk_type": c.chunk_type, "source": c.source,
                             "chunk_id": c.chunk_id},
                )
                for c, v in zip(batch, batch_vecs)
            ]
            self._qdrant.upsert(collection_name=self._col, points=points)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 15, language: str | None = None) -> list[CodeChunk]:
        vec = self._embed([query])[0]
        filt = None
        if language:
            filt = models.Filter(must=[models.FieldCondition(
                key="language", match=models.MatchValue(value=language))])
        results = self._qdrant.search(
            collection_name=self._col,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
            query_filter=filt,
        )
        return [self._to_chunk(r) for r in results]

    def scroll_all(self, limit: int = 10_000) -> list[CodeChunk]:
        """Return all indexed chunks (for BM25 index construction)."""
        records, _ = self._qdrant.scroll(
            collection_name=self._col, limit=limit, with_payload=True, with_vectors=False
        )
        return [self._to_chunk(r) for r in records]

    def count(self) -> int:
        return self._qdrant.count(self._col).count

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        def _call():
            return self._oai.embeddings.create(
                model=settings.embedding_model, input=texts, dimensions=self._dim
            )
        resp = call_with_backoff(_call)
        return [item.embedding for item in resp.data]

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        resp = await self._aoai.embeddings.create(
            model=settings.embedding_model, input=texts, dimensions=self._dim
        )
        return [item.embedding for item in resp.data]

    @staticmethod
    def _to_chunk(r) -> CodeChunk:  # type: ignore[no-untyped-def]
        p = r.payload or {}
        return CodeChunk(
            file_path=p.get("file_path", ""),
            start_line=p.get("start_line", 0),
            end_line=p.get("end_line", 0),
            language=p.get("language", ""),
            chunk_type=p.get("chunk_type", "module"),
            source=p.get("source", ""),
            relevance_score=float(getattr(r, "score", 0.0)),
        )
