"""AST-aware code chunker. Splits at function/class boundaries; falls back to sliding window."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
}

CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python":     {"function_definition", "async_function_definition", "class_definition"},
    "javascript": {"function_declaration", "arrow_function", "method_definition", "class_declaration"},
    "typescript": {"function_declaration", "arrow_function", "method_definition",
                   "class_declaration", "interface_declaration", "type_alias_declaration"},
    "go":         {"function_declaration", "method_declaration", "type_declaration"},
}

SKIP_DIRS: set[str] = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".ruff_cache", "migrations",
}
SKIP_EXTS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".whl", ".pyc", ".so", ".dylib", ".dll",
    ".lock", ".sum",
}


@dataclass
class RawChunk:
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    source: str
    chunk_id: str


class ASTChunker:
    def __init__(self) -> None:
        self._parsers: dict[str, object] = {}
        self._node_types: dict[str, set[str]] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        for lang, pkg in [
            ("python", "tree_sitter_python"),
            ("javascript", "tree_sitter_javascript"),
            ("typescript", "tree_sitter_typescript"),
            ("go", "tree_sitter_go"),
        ]:
            try:
                mod = __import__(pkg, fromlist=["language"])
                from tree_sitter import Language, Parser
                parser = Parser(Language(mod.language()))
                self._parsers[lang] = parser
                self._node_types[lang] = CHUNK_NODE_TYPES[lang]
                log.debug("grammar_loaded", language=lang)
            except Exception as exc:
                log.warning("grammar_unavailable", language=lang, error=str(exc))

    def chunk_repository(self, repo_root: str) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        root = Path(repo_root)
        file_count = 0
        for path in root.rglob("*"):
            if any(p in SKIP_DIRS for p in path.parts):
                continue
            if not path.is_file() or path.suffix in SKIP_EXTS:
                continue
            lang = LANGUAGE_MAP.get(path.suffix.lower())
            if lang is None:
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                rel = str(path.relative_to(root))
                chunks.extend(self._chunk_file(source, rel, lang))
                file_count += 1
            except OSError:
                continue
        log.info("repo_chunked", files=file_count, chunks=len(chunks), root=repo_root)
        return chunks

    def _chunk_file(self, source: str, rel_path: str, lang: str) -> list[RawChunk]:
        parser = self._parsers.get(lang)
        if parser is None:
            return self._sliding_window(source, rel_path, lang)
        lines = source.splitlines(keepends=True)
        tree = parser.parse(source.encode())  # type: ignore[attr-defined]
        chunks: list[RawChunk] = []
        self._walk(tree.root_node, lines, rel_path, lang, self._node_types[lang], chunks)
        if not chunks:
            chunks.append(self._make(source, rel_path, lang, "module", 1, len(lines)))
        return chunks

    def _walk(self, node, lines, rel_path, lang, targets, out) -> None:  # type: ignore[no-untyped-def]
        if node.type in targets:
            s, e = node.start_point[0], node.end_point[0]
            label = "class" if "class" in node.type else "function"
            out.append(self._make("".join(lines[s:e + 1]), rel_path, lang, label, s + 1, e + 1))
            return
        for child in node.children:
            self._walk(child, lines, rel_path, lang, targets, out)

    @staticmethod
    def _make(source: str, path: str, lang: str, ctype: str, s: int, e: int) -> RawChunk:
        return RawChunk(
            file_path=path, start_line=s, end_line=e,
            language=lang, chunk_type=ctype, source=source,
            chunk_id=hashlib.sha256(source.encode()).hexdigest()[:16],
        )

    def _sliding_window(self, source: str, path: str, lang: str,
                        window: int = 500, overlap: int = 50) -> list[RawChunk]:
        char_w, char_o = window * 4, overlap * 4
        chunks, pos = [], 0
        while pos < len(source):
            end = min(pos + char_w, len(source))
            snippet = source[pos:end]
            sl = source[:pos].count("\n") + 1
            el = sl + snippet.count("\n")
            chunks.append(self._make(snippet, path, lang, "module", sl, el))
            pos += char_w - char_o
        return chunks
