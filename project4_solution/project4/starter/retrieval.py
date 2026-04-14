"""
Sparse Retrieval Baselines — TF-IDF and BM25  (Task 4 — IMPLEMENTED)
=====================================================================
Both methods rely on LEXICAL overlap between query and documents.
They will score poorly on queries like:
  "heart attack treatment aspirin"
against documents containing only:
  "myocardial infarction antiplatelet therapy clopidogrel"

Optimisations for subsequent queries:
  - TF-IDF: document matrix stored in memory after fit(); only the tiny
    query vector is recomputed per search (~0.5 ms for 2000 docs).
  - BM25:   BM25Okapi object cached in memory; get_scores() is vectorised.
  - Both:   optionally save/load fitted index to/from disk (joblib).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import RetrievalMethod, RetrievalResponse, SearchResult


# ---------------------------------------------------------------------------
# TF-IDF Retriever
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """
    Classic TF-IDF cosine similarity retrieval.

    Strengths  : fast, no GPU, interpretable, very low subsequent-query latency
    Weaknesses : vocabulary mismatch (exact token overlap required)

    Subsequent-query speed-up:
      The document TF-IDF matrix is stored in RAM after fit().
      Each search only transforms ONE query vector then does a sparse
      dot product — typically < 2 ms for 2000 documents.

    Disk caching:
      Call save(path) / load(path) to persist the fitted vectorizer and
      document matrix so that the next run skips re-fitting entirely.
    """

    def __init__(
        self,
        max_features:    int = 30_000,
        ngram_range:     tuple[int, int] = (1, 2),
        sublinear_tf:    bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
        )
        self._doc_matrix  = None
        self._doc_ids:    list[str]  = []
        self._doc_titles: list[str]  = []
        self._doc_topics: list[str]  = []
        self._fitted      = False

        # Timing log for comparison
        self._query_times: list[float] = []

    def fit(self, docs: pd.DataFrame, cache_path: Optional[Path] = None) -> None:
        """
        Build TF-IDF matrix.

        Args:
            docs:       DataFrame with [doc_id, full_text, title, topic].
            cache_path: if given, save/load the fitted index at this path.
        """
        if cache_path and cache_path.exists():
            print(f"[TF-IDF] Loading cached index from {cache_path} ...")
            self.load(cache_path)
            return

        t0 = time.perf_counter()
        self._doc_ids    = docs["doc_id"].tolist()
        self._doc_titles = docs["title"].tolist()
        self._doc_topics = docs["topic"].tolist() if "topic" in docs.columns else [""] * len(docs)
        self._doc_matrix = self.vectorizer.fit_transform(docs["full_text"].fillna(""))
        self._fitted     = True
        elapsed = time.perf_counter() - t0

        print(f"[TF-IDF] Index built in {elapsed:.2f}s — "
              f"{self._doc_matrix.shape[0]} docs, "
              f"{self._doc_matrix.shape[1]} features")

        if cache_path:
            self.save(cache_path)

    def search(
        self,
        query_id:   str,
        query:      str,
        top_k:      int = 10,
    ) -> RetrievalResponse:
        """
        Search by cosine similarity.

        First query: ~2–5 ms (vectorizer transform + sparse matmul).
        Subsequent queries: same cost — no re-fitting needed.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before search()")

        t0       = time.perf_counter()
        q_vec    = self.vectorizer.transform([query])
        scores   = cosine_similarity(q_vec, self._doc_matrix)[0]
        top_idx  = np.argsort(scores)[::-1][:top_k]
        elapsed  = (time.perf_counter() - t0) * 1000
        self._query_times.append(elapsed)

        results = [
            SearchResult(
                rank=rank + 1,
                doc_id=self._doc_ids[i],
                score=float(scores[i]),
                topic=self._doc_topics[i],
                title=self._doc_titles[i],
                retrieval_method=RetrievalMethod.TFIDF,
            )
            for rank, i in enumerate(top_idx)
        ]

        return RetrievalResponse(
            query_id=query_id,
            original_query=query,
            results=results,
            retrieval_method=RetrievalMethod.TFIDF,
            latency_ms=elapsed,
            n_results=len(results),
        )

    def save(self, path: Path) -> None:
        """Persist vectorizer + matrix to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer":  self.vectorizer,
                "doc_matrix":  self._doc_matrix,
                "doc_ids":     self._doc_ids,
                "doc_titles":  self._doc_titles,
                "doc_topics":  self._doc_topics,
            }, f)
        print(f"[TF-IDF] Index saved -> {path}")

    def load(self, path: Path) -> None:
        """Restore fitted index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer    = data["vectorizer"]
        self._doc_matrix   = data["doc_matrix"]
        self._doc_ids      = data["doc_ids"]
        self._doc_titles   = data["doc_titles"]
        self._doc_topics   = data["doc_topics"]
        self._fitted       = True
        print(f"[TF-IDF] Index loaded from {path} — "
              f"{self._doc_matrix.shape[0]} docs")

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0

    def avg_query_time_ms(self) -> float:
        return float(np.mean(self._query_times)) if self._query_times else 0.0


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 (Okapi BM25) — improved term-frequency-based retrieval.

    BM25 addresses TF saturation and document length normalisation,
    but still requires LEXICAL overlap — no semantic understanding.

    Strengths  : better than TF-IDF for long documents; standard baseline in IR
    Weaknesses : vocabulary mismatch equally severe as TF-IDF

    Subsequent-query speed-up:
      BM25Okapi stores pre-computed IDF weights and term frequencies.
      Each search is a vectorised numpy operation — typically < 5 ms
      for 2000 documents regardless of how many times it is called.

    Disk caching:
      Call save(path) / load(path) to persist the fitted BM25 object.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1   = k1
        self.b    = b
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids:    list[str] = []
        self._doc_titles: list[str] = []
        self._doc_topics: list[str] = []
        self._query_times: list[float] = []

    def fit(self, docs: pd.DataFrame, cache_path: Optional[Path] = None) -> None:
        if cache_path and cache_path.exists():
            print(f"[BM25] Loading cached index from {cache_path} ...")
            self.load(cache_path)
            return

        t0 = time.perf_counter()
        self._doc_ids    = docs["doc_id"].tolist()
        self._doc_titles = docs["title"].tolist()
        self._doc_topics = docs["topic"].tolist() if "topic" in docs.columns else [""] * len(docs)

        tokenised = [
            text.lower().split()
            for text in docs["full_text"].fillna("")
        ]
        self._bm25 = BM25Okapi(tokenised, k1=self.k1, b=self.b)
        elapsed = time.perf_counter() - t0

        print(f"[BM25] Index built in {elapsed:.2f}s — {len(docs)} docs")

        if cache_path:
            self.save(cache_path)

    def search(
        self,
        query_id: str,
        query:    str,
        top_k:    int = 10,
    ) -> RetrievalResponse:
        if self._bm25 is None:
            raise RuntimeError("Call fit() before search()")

        t0      = time.perf_counter()
        tokens  = query.lower().split()
        scores  = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        self._query_times.append(elapsed)

        results = [
            SearchResult(
                rank=rank + 1,
                doc_id=self._doc_ids[i],
                score=float(scores[i]),
                topic=self._doc_topics[i],
                title=self._doc_titles[i],
                retrieval_method=RetrievalMethod.BM25,
            )
            for rank, i in enumerate(top_idx)
        ]

        return RetrievalResponse(
            query_id=query_id,
            original_query=query,
            results=results,
            retrieval_method=RetrievalMethod.BM25,
            latency_ms=elapsed,
            n_results=len(results),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25":       self._bm25,
                "doc_ids":    self._doc_ids,
                "doc_titles": self._doc_titles,
                "doc_topics": self._doc_topics,
                "k1":         self.k1,
                "b":          self.b,
            }, f)
        print(f"[BM25] Index saved -> {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25       = data["bm25"]
        self._doc_ids    = data["doc_ids"]
        self._doc_titles = data["doc_titles"]
        self._doc_topics = data["doc_topics"]
        self.k1          = data["k1"]
        self.b           = data["b"]
        print(f"[BM25] Index loaded from {path} — {len(self._doc_ids)} docs")

    def avg_query_time_ms(self) -> float:
        return float(np.mean(self._query_times)) if self._query_times else 0.0


# ---------------------------------------------------------------------------
# Vocabulary overlap analysis  (Task 4a support)
# ---------------------------------------------------------------------------

def analyse_vocabulary_mismatch(
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each query, compute the fraction of query tokens present in the
    corpus vocabulary and in the relevant documents specifically.

    Returns a DataFrame with per-query mismatch statistics.
    """
    corpus_vocab: set[str] = set()
    for text in corpus_df["full_text"].fillna(""):
        for tok in text.lower().split():
            corpus_vocab.add(tok.strip(".,;:()[]"))

    rows = []
    for _, qrow in queries_df.iterrows():
        q_tokens    = qrow["query_text"].lower().split()
        q_tokens    = [t.strip(".,;:()[]") for t in q_tokens]
        n_overlap   = sum(1 for t in q_tokens if t in corpus_vocab)
        lay_terms   = [t.strip() for t in str(qrow.get("lay_terms", "")).split(";") if t.strip()]
        n_lay_in_corpus = sum(
            1 for term in lay_terms
            if any(w in corpus_vocab for w in term.lower().split())
        )
        rows.append({
            "query_id":         qrow["query_id"],
            "query_text":       qrow["query_text"],
            "n_query_tokens":   len(q_tokens),
            "n_overlap_corpus": n_overlap,
            "overlap_pct":      round(100 * n_overlap / max(1, len(q_tokens)), 1),
            "n_lay_terms":      len(lay_terms),
            "lay_terms_in_corpus": n_lay_in_corpus,
            "lay_mismatch_pct": round(100 * (1 - n_lay_in_corpus / max(1, len(lay_terms))), 1),
        })

    df = pd.DataFrame(rows)
    print(f"\nVocabulary mismatch analysis ({len(df)} queries):")
    print(f"  Avg query-corpus token overlap : {df['overlap_pct'].mean():.1f}%")
    print(f"  Queries where lay term absent  : "
          f"{(df['lay_mismatch_pct'] == 100).sum()} / {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    data_dir = Path(__file__).parent.parent / "data"
    corpus_path = data_dir / "clinical_corpus.csv"

    if not corpus_path.exists():
        print("Run: python data/generate_dataset.py first")
        sys.exit(1)

    corpus_df = pd.read_csv(corpus_path)
    print(f"Corpus: {len(corpus_df)} docs")

    # --- TF-IDF ---
    tfidf = TFIDFRetriever()
    tfidf.fit(corpus_df)

    demo_q = "heart attack treatment aspirin antiplatelet"
    for i in range(3):
        resp = tfidf.search(f"Q-demo-{i}", demo_q, top_k=5)
        print(f"[TF-IDF] Query #{i+1} latency: {resp.latency_ms:.2f} ms")

    print(f"  Average TF-IDF latency: {tfidf.avg_query_time_ms():.2f} ms")

    # --- BM25 ---
    bm25 = BM25Retriever()
    bm25.fit(corpus_df)

    for i in range(3):
        resp = bm25.search(f"Q-demo-{i}", demo_q, top_k=5)
        print(f"[BM25]   Query #{i+1} latency: {resp.latency_ms:.2f} ms")

    print(f"  Average BM25 latency: {bm25.avg_query_time_ms():.2f} ms")

    # Note the vocabulary mismatch
    print(f"\nNote: 'heart attack' ≠ 'myocardial infarction' for these lexical methods.")
    print("After query expansion, BM25 should rank STEMI/MI documents much higher.")
