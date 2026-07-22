"""A small, dependency-light TF-IDF retriever for the RAG corpus.

Financial risk narratives are timeliness-sensitive: a rating downgrade from
six months ago matters more than a routine announcement from last week's
close relevance score alone would suggest. So retrieval combines two
signals — semantic (TF-IDF cosine) relevance and a recency weight — rather
than ranking on text similarity alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import jieba
import numpy as np

from fin_risk.data.loader import Document

STOPWORDS = {
    "的", "了", "在", "是", "和", "与", "及", "对", "为", "将", "已", "其",
    "此", "该", "并", "等", "也", "而", "但", "又", "或", "之", "上", "下",
    "一", "个", "有", "被", "使", "由", "至", "较", "更", "再", "就", "还",
}


def _tokenize(text: str) -> list[str]:
    return [tok for tok in jieba.lcut(text) if tok.strip() and tok not in STOPWORDS]


@dataclass
class RetrievalResult:
    document: Document
    similarity: float
    recency_weight: float
    combined_score: float


class TfidfRetriever:
    """Builds a TF-IDF index over a fixed document set and ranks by
    (1 - recency_weight) * cosine_similarity + recency_weight * recency."""

    def __init__(self, documents: list[Document], recency_half_life_days: float = 120.0):
        self.documents = documents
        self.recency_half_life_days = recency_half_life_days
        self._doc_tokens = [_tokenize(f"{d.title} {d.content}") for d in documents]
        vocab = sorted({tok for toks in self._doc_tokens for tok in toks})
        self._vocab_index = {tok: i for i, tok in enumerate(vocab)}
        self._idf = np.ones(len(vocab))
        self._doc_vectors = np.zeros((len(documents), len(vocab)))
        if documents and vocab:
            self._build_index()

    def _build_index(self) -> None:
        n_docs = len(self.documents)
        n_vocab = len(self._vocab_index)
        tf = np.zeros((n_docs, n_vocab))
        for i, toks in enumerate(self._doc_tokens):
            for tok in toks:
                tf[i, self._vocab_index[tok]] += 1
            if toks:
                tf[i] /= len(toks)
        df = (tf > 0).sum(axis=0)
        self._idf = np.log((n_docs + 1) / (df + 1)) + 1
        tfidf = tf * self._idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._doc_vectors = tfidf / norms

    def _vectorize_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab_index))
        toks = _tokenize(query)
        for tok in toks:
            idx = self._vocab_index.get(tok)
            if idx is not None:
                vec[idx] += 1
        if toks:
            vec /= len(toks)
        vec *= self._idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(
        self,
        query: str,
        company_id: Optional[str] = None,
        top_k: int = 5,
        recency_boost: float = 0.2,
        reference_date: Optional[str] = None,
    ) -> list[RetrievalResult]:
        if not self.documents or not self._vocab_index:
            return []

        candidate_idx = [
            i for i, d in enumerate(self.documents) if company_id is None or d.company_id == company_id
        ]
        if not candidate_idx:
            return []

        ref = (
            datetime.fromisoformat(reference_date)
            if reference_date
            else max(datetime.fromisoformat(self.documents[i].date) for i in candidate_idx)
        )

        q_vec = self._vectorize_query(query)
        results: list[RetrievalResult] = []
        for i in candidate_idx:
            doc = self.documents[i]
            similarity = float(self._doc_vectors[i] @ q_vec)
            age_days = max((ref - datetime.fromisoformat(doc.date)).days, 0)
            recency_weight = math.exp(-age_days / self.recency_half_life_days)
            combined = (1 - recency_boost) * similarity + recency_boost * recency_weight
            results.append(RetrievalResult(doc, similarity, recency_weight, combined))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]
