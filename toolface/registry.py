from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from ..schemas import ToolFunction, ToolSchema


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is",
    "are", "be", "with", "on", "by", "at", "as", "from", "this",
    "that", "it", "its", "into", "any", "all", "we", "you",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")
            if t.lower() not in _STOP and len(t) > 1]


class ToolFace:

    def __init__(self):
        self._schemas: Dict[str, ToolSchema] = {}
        self._functions: Dict[str, ToolFunction] = {}
        self._df: Dict[str, int] = defaultdict(int)
        self._tf: Dict[str, Counter] = {}
        self._n_docs: int = 0
        self._embed_fn: Optional[Callable[[List[str]], "list"]] = None
        self._embeddings: Dict[str, list] = {}

    def register(self, schema: ToolSchema, fn: Callable[..., object]) -> None:
        if schema.id in self._schemas:
            raise ValueError(f"Tool id collision: {schema.id}")
        self._schemas[schema.id] = schema
        self._functions[schema.id] = ToolFunction(schema_id=schema.id, fn=fn)
        doc = self._doc_text(schema)
        tokens = tokenize(doc)
        tf = Counter(tokens)
        self._tf[schema.id] = tf
        for term in tf:
            self._df[term] += 1
        self._n_docs += 1
        if self._embed_fn is not None:
            self._embeddings[schema.id] = self._embed_fn([doc])[0]

    def _doc_text(self, s: ToolSchema) -> str:
        param_blurb = " ".join(f"{p.name} {p.description}" for p in s.parameters)
        return f"{s.id} {s.name} {s.description} {s.category} {param_blurb} {s.returns}"

    def enable_embeddings(self, embed_fn: Callable[[List[str]], list]) -> None:
        self._embed_fn = embed_fn
        ids = list(self._schemas.keys())
        if ids:
            docs = [self._doc_text(self._schemas[i]) for i in ids]
            for sid, vec in zip(ids, embed_fn(docs)):
                self._embeddings[sid] = vec

    def get_schema(self, tool_id: str) -> ToolSchema:
        if tool_id not in self._schemas:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return self._schemas[tool_id]

    def get_function(self, tool_id: str) -> ToolFunction:
        if tool_id not in self._functions:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return self._functions[tool_id]

    def __len__(self) -> int:
        return len(self._schemas)

    def list_ids(self) -> List[str]:
        return list(self._schemas)

    def list_schemas(self) -> List[ToolSchema]:
        return list(self._schemas.values())

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        alpha: float = 0.5,
    ) -> List[Tuple[ToolSchema, float]]:
        if not self._schemas:
            return []

        candidate_ids = [
            sid for sid, s in self._schemas.items()
            if (category is None or s.category == category)
            and (source is None or s.source == source)
        ]
        if not candidate_ids:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            lex = {sid: 0.0 for sid in candidate_ids}
        else:
            q_tf = Counter(q_tokens)
            q_vec = {t: tf * self._idf(t) for t, tf in q_tf.items()}
            q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
            lex = {}
            for sid in candidate_ids:
                d_tf = self._tf[sid]
                dot = 0.0
                d_norm_sq = 0.0
                for t, tf in d_tf.items():
                    w = tf * self._idf(t)
                    d_norm_sq += w * w
                    if t in q_vec:
                        dot += w * q_vec[t]
                d_norm = math.sqrt(d_norm_sq) or 1.0
                lex[sid] = dot / (q_norm * d_norm)

        if self._embed_fn is not None and self._embeddings:
            q_emb = self._embed_fn([query])[0]
            qn = math.sqrt(sum(x * x for x in q_emb)) or 1.0
            dense = {}
            for sid in candidate_ids:
                v = self._embeddings.get(sid)
                if not v:
                    dense[sid] = 0.0
                    continue
                vn = math.sqrt(sum(x * x for x in v)) or 1.0
                dense[sid] = sum(a * b for a, b in zip(q_emb, v)) / (qn * vn)
            scores = {sid: alpha * dense[sid] + (1 - alpha) * lex[sid]
                      for sid in candidate_ids}
        else:
            scores = lex

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self._schemas[sid], score) for sid, score in ranked]

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0
