"""
Multi-Stage Retrieval Pipeline  (Task 11 — IMPLEMENTED)
=========================================================
Combines query expansion, sparse first-stage retrieval, and dense re-ranking.

Architecture:
  Stage 1: Query Expansion
    - Map lay terms -> clinical synonyms using MEDICAL_ONTOLOGY
    - Produces enriched query for Stage 2 (BM25)

  Stage 2: BM25 Candidate Retrieval  (fast, sparse)
    - Retrieve top-100 candidates using expanded query
    - ~1-5 ms on CPU

  Stage 3: SBERT Re-ranking  (accurate, dense)
    - Re-score top-100 candidates with cosine similarity
    - ~10-30 ms on CPU
    - Returns top-K final results

Speed-up strategy for subsequent queries:
  - BM25 index stays in RAM: fast candidate retrieval each time.
  - Doc lookup table (doc_id -> full_text) pre-built at index_corpus() time.
  - SBERT model loaded once; only the 100 candidate texts + 1 query string
    are encoded per query (vs 2000 docs for full dense retrieval).
  - Total pipeline latency per query: ~15-50 ms on CPU.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from embeddings import DenseRetriever, EmbeddingEncoder
from query_expansion import QueryExpander
from retrieval import BM25Retriever
from schemas import RetrievalMethod, RetrievalResponse, SearchResult


# ---------------------------------------------------------------------------
# Multi-Stage Pipeline
# ---------------------------------------------------------------------------

class MultiStagePipeline:
    """
    Three-stage retrieval: Expansion -> BM25 candidates -> SBERT re-rank.

    Args:
        expander:       QueryExpander instance (pass None to skip expansion).
        bm25:           Fitted BM25Retriever for first-stage candidate recall.
        encoder:        EmbeddingEncoder for dense re-ranking.
        candidate_k:    Number of candidates from BM25 (default: 100).
        final_k:        Number of final results after SBERT re-rank (default: 10).
    """

    def __init__(
        self,
        expander:       Optional[QueryExpander],
        bm25:           BM25Retriever,
        encoder:        EmbeddingEncoder,
        candidate_k:    int = 100,
        final_k:        int = 10,
        doc_embeddings: Optional[np.ndarray] = None,
        doc_id_to_idx:  Optional[dict[str, int]] = None,
    ) -> None:
        self.expander       = expander
        self.bm25           = bm25
        self.encoder        = encoder
        self.candidate_k    = candidate_k
        self.final_k        = final_k
        self.doc_embeddings = doc_embeddings  # pre-computed (N, D) matrix
        self.doc_id_to_idx  = doc_id_to_idx or {}  # doc_id -> row index in matrix

        # doc_id -> {full_text, title, topic}
        self._doc_lookup: dict[str, dict] = {}
        self._query_times: list[float] = []

    def index_corpus(
        self,
        corpus_df: pd.DataFrame,
        doc_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build BM25 index and doc lookup table from corpus DataFrame.

        If doc_embeddings is provided, store them for fast SBERT re-ranking.

        Called once at startup. Subsequent queries reuse both in memory.
        """
        self.bm25.fit(corpus_df)
        self.doc_id_to_idx = {}
        for idx, row in corpus_df.iterrows():
            doc_id = row["doc_id"]
            self._doc_lookup[doc_id] = {
                "full_text": str(row.get("full_text", "")),
                "title":     str(row.get("title", "")),
                "topic":     str(row.get("topic", "")),
            }
            if doc_embeddings is not None:
                self.doc_id_to_idx[doc_id] = idx
        if doc_embeddings is not None:
            self.doc_embeddings = doc_embeddings
        print(f"[Pipeline] Corpus indexed: {len(self._doc_lookup)} docs")

    # ------------------------------------------------------------------
    # Task 11: IMPLEMENTED — retrieve()
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_id:  str,
        query:     str,
        top_k:     Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Run the full multi-stage pipeline for a single query.

        Step 1 — Query Expansion (if self.expander is not None):
            Expands lay terms to clinical synonyms for BM25.

        Step 2 — BM25 candidate retrieval:
            Uses expanded query to retrieve self.candidate_k candidates.
            BM25 is great at recall — we cast a wide net here.

        Step 3 — SBERT re-ranking:
            Encodes candidate full texts + original query as dense vectors.
            Re-scores by cosine similarity and keeps top final_k.
            We use the ORIGINAL (un-expanded) query here because SBERT
            already understands semantics — expansion only helps BM25.

        Step 4 — Return RetrievalResponse with full provenance.
        """
        if top_k is None:
            top_k = self.final_k

        t_start = time.perf_counter()

        # ---- Stage 1: Query Expansion ----
        if self.expander is not None:
            expansion = self.expander.expand(query)
            search_query = expansion.expansion_query
        else:
            expansion    = None
            search_query = query

        # ---- Stage 2: BM25 Candidate Retrieval ----
        candidates = self.bm25.search(
            query_id, search_query, top_k=self.candidate_k
        )
        candidate_doc_ids = [r.doc_id for r in candidates.results]

        # Gather full texts for candidate documents
        candidate_texts  = []
        candidate_titles = []
        candidate_topics = []
        for doc_id in candidate_doc_ids:
            info = self._doc_lookup.get(doc_id, {})
            candidate_texts.append(info.get("full_text", ""))
            candidate_titles.append(info.get("title", ""))
            candidate_topics.append(info.get("topic", ""))

        # ---- Stage 3: SBERT Re-ranking ----
        # Use pre-computed embeddings when available (avoid redundant encoding)
        if self.doc_embeddings is not None and len(self.doc_id_to_idx) > 0:
            # Look up pre-computed embeddings for candidate documents
            cand_indices = [self.doc_id_to_idx.get(d, -1) for d in candidate_doc_ids]
            valid_mask = [i >= 0 for i in cand_indices]
            valid_indices = [i for i, v in zip(cand_indices, valid_mask) if v]

            if valid_indices:
                doc_vecs = self.doc_embeddings[valid_indices]
                # Re-filter candidate data to only those with valid embeddings
                candidate_doc_ids = [d for d, v in zip(candidate_doc_ids, valid_mask) if v]
                candidate_titles = [t for t, v in zip(candidate_titles, valid_mask) if v]
                candidate_topics = [t for t, v in zip(candidate_topics, valid_mask) if v]
            else:
                # Fallback: encode from scratch
                doc_vecs = self.encoder.encode(
                    candidate_texts, show_progress=False, normalize=True
                )
        else:
            # No pre-computed embeddings — encode from scratch
            doc_vecs = self.encoder.encode(
                candidate_texts, show_progress=False, normalize=True
            )  # shape: (candidate_k, D)

        # Encode original query (NOT expanded — SBERT handles semantics natively)
        q_vec = self.encoder.encode_single(query, normalize=True)  # shape: (D,)

        # Cosine similarity (vectors are L2-normalised -> dot product == cosine)
        rerank_scores = doc_vecs @ q_vec   # shape: (candidate_k,)

        # Sort descending, keep top_k
        sorted_idx = np.argsort(rerank_scores)[::-1][:top_k]

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        self._query_times.append(latency_ms)

        # ---- Build SearchResult list ----
        results = []
        for rank, idx in enumerate(sorted_idx, start=1):
            results.append(SearchResult(
                rank=rank,
                doc_id=candidate_doc_ids[idx],
                score=float(rerank_scores[idx]),
                topic=candidate_topics[idx],
                title=candidate_titles[idx],
                retrieval_method=RetrievalMethod.MULTISTAGE,
            ))

        return RetrievalResponse(
            query_id=query_id,
            original_query=query,
            expansion=expansion,
            results=results,
            retrieval_method=RetrievalMethod.MULTISTAGE,
            latency_ms=latency_ms,
            n_results=len(results),
        )

    def avg_query_time_ms(self) -> float:
        return float(np.mean(self._query_times)) if self._query_times else 0.0


# ---------------------------------------------------------------------------
# Ablation runner (Task 12 support — runs all 4 conditions)
# ---------------------------------------------------------------------------

def run_ablation(
    corpus_df:    pd.DataFrame,
    queries_df:   pd.DataFrame,
    top_k:        int = 10,
    verbose:      bool = True,
    cache_dir:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Run four retrieval conditions on all test queries and compute MRR@K.

    Conditions:
      A. BM25 only (no expansion)        <- vocabulary mismatch baseline
      B. BM25 + query expansion          <- shows value of expansion alone
      C. SBERT dense (no expansion)      <- semantic retrieval
      D. MultiStage (exp + BM25 + SBERT) <- full production pipeline

    Expected ranking: D > C >= B > A
    Expected MRR improvement A->D: from ~0.15 to ~0.65+

    Returns:
        DataFrame with MRR@K and Recall@K for each condition.
    """
    from evaluate import mrr_at_k, recall_at_k
    from embeddings import DenseRetriever, EmbeddingEncoder

    cache_path = None
    if cache_dir:
        from pathlib import Path
        cache_path = Path(cache_dir) / "corpus_embeddings.npy"

    # ---- Setup ----
    bm25_plain = BM25Retriever()
    bm25_plain.fit(corpus_df)

    bm25_exp = BM25Retriever()
    bm25_exp.fit(corpus_df)

    expander = QueryExpander()

    encoder = EmbeddingEncoder()
    dense   = DenseRetriever(encoder)

    try:
        dense.fit(corpus_df, cache_path=cache_path)
        dense_available = True
    except Exception as e:
        dense_available = False
        if verbose:
            print(f"  SBERT unavailable ({e}) — skipping dense conditions")

    pipeline = MultiStagePipeline(expander, bm25_exp, encoder, candidate_k=100)
    if dense_available:
        pipeline.index_corpus(corpus_df, doc_embeddings=dense._doc_matrix)
    else:
        pipeline.index_corpus(corpus_df)

    # ---- Evaluate helper ----
    results_agg: dict[str, list] = {"condition": [], "mrr_at_k": [], "recall_at_k": []}

    def _evaluate_condition(responses, name):
        mrr_scores, rec_scores = [], []
        for resp, (_, qrow) in zip(responses, queries_df.iterrows()):
            relevant_topics = set(qrow["relevant_topics"].split("; "))
            relevance = [
                1 if (r.topic in relevant_topics) else 0
                for r in resp.results
            ]
            mrr_scores.append(mrr_at_k(relevance, top_k))
            rec_scores.append(recall_at_k(relevance, top_k))
        results_agg["condition"].append(name)
        results_agg["mrr_at_k"].append(round(float(np.mean(mrr_scores)), 4))
        results_agg["recall_at_k"].append(round(float(np.mean(rec_scores)), 4))
        if verbose:
            print(f"  {name:<38} MRR@{top_k}={results_agg['mrr_at_k'][-1]:.4f}  "
                  f"Recall@{top_k}={results_agg['recall_at_k'][-1]:.4f}")

    if verbose:
        print(f"\nAblation study (top_k={top_k}):")

    # Condition A — BM25 only
    t0 = time.perf_counter()
    responses_a = [
        bm25_plain.search(row["query_id"], row["query_text"], top_k=top_k)
        for _, row in queries_df.iterrows()
    ]
    if verbose:
        print(f"  [A] BM25 wall time: {(time.perf_counter()-t0)*1000:.0f} ms total")
    _evaluate_condition(responses_a, "A. BM25 (no expansion)")

    # Condition B — BM25 + expansion
    t0 = time.perf_counter()
    responses_b = []
    for _, row in queries_df.iterrows():
        exp = expander.expand(row["query_text"])
        resp = bm25_exp.search(row["query_id"], exp.expansion_query, top_k=top_k)
        responses_b.append(resp)
    if verbose:
        print(f"  [B] BM25+Exp wall time: {(time.perf_counter()-t0)*1000:.0f} ms total")
    _evaluate_condition(responses_b, "B. BM25 + expansion")

    # Condition C — SBERT dense
    if dense_available:
        t0 = time.perf_counter()
        responses_c = [
            dense.search(row["query_id"], row["query_text"], top_k=top_k)
            for _, row in queries_df.iterrows()
        ]
        if verbose:
            print(f"  [C] SBERT wall time: {(time.perf_counter()-t0)*1000:.0f} ms total")
        _evaluate_condition(responses_c, "C. SBERT dense")
    else:
        if verbose:
            print("  C. SBERT dense   (skipped — encode() unavailable)")

    # Condition D — MultiStage
    if dense_available:
        t0 = time.perf_counter()
        responses_d = [
            pipeline.retrieve(row["query_id"], row["query_text"])
            for _, row in queries_df.iterrows()
        ]
        if verbose:
            print(f"  [D] MultiStage wall time: {(time.perf_counter()-t0)*1000:.0f} ms total")
        _evaluate_condition(responses_d, "D. MultiStage (full pipeline)")
    else:
        if verbose:
            print("  D. MultiStage    (skipped — SBERT unavailable)")

    return pd.DataFrame(results_agg)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    data_dir    = Path(__file__).parent.parent / "data"
    corpus_path = data_dir / "clinical_corpus.csv"
    query_path  = data_dir / "query_testset.csv"

    if not corpus_path.exists():
        print("Run: python data/generate_dataset.py first")
        sys.exit(1)

    corpus_df = pd.read_csv(corpus_path)
    query_df  = pd.read_csv(query_path)

    ablation_df = run_ablation(corpus_df, query_df, top_k=10)
    print("\nAblation summary:")
    print(ablation_df.to_string(index=False))
