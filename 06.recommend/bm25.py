"""BM25Okapi，用于意象文档排序（无第三方 rank_bm25 依赖）。"""

from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = (
            sum(len(d) for d in corpus) / self.corpus_size if self.corpus_size else 0.0
        )
        self.doc_freqs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}

        df: dict[str, int] = {}
        for doc in corpus:
            self.doc_len.append(len(doc))
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for w in freqs:
                df[w] = df.get(w, 0) + 1

        for word, freq in df.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i]
            score = 0.0
            for q in query_tokens:
                if q not in freqs:
                    continue
                idf = self.idf.get(q, 0.0)
                f = freqs[q]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * f * (self.k1 + 1) / denom
            scores[i] = score
        return scores
