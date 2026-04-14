"""
FAISS Index Builders — IVFFlat, HNSW, IVF-PQ  (Tasks 7-10 IMPLEMENTED)
========================================================================
Approximate Nearest Neighbour (ANN) search for large-scale retrieval.

At 10 million clinical notes, exact cosine search is infeasible:
  - Memory: 10M x 384 x 4 bytes = 14.4 GB (exceeds 16 GB RAM)
  - Latency: ~100-500 ms per query on CPU (exceeds 50 ms SLA)

FAISS trades a small amount of recall for massive speed and memory gains.

Index comparison (at 10M docs):
+-----------+------------------+-------------------+-------------------+----------+
| Index     | Memory @10M docs | Query latency CPU | Recall@10 typical | Suitable |
+-----------+------------------+-------------------+-------------------+----------+
| Flat      | 14.4 GB          | ~200 ms           | 100% (exact)      | Dev only |
| IVFFlat   | ~15 GB           | ~20-40 ms         | ~95%              | Medium   |
| HNSW      | ~18 GB           | ~2-5 ms           | ~98%              | No (RAM) |
| IVF-PQ    | ~0.9 GB          | ~5-15 ms          | ~85-90%           | YES      |
+-----------+------------------+-------------------+-------------------+----------+

Speed-up strategy:
  - FAISS index built once from pre-computed embeddings.
  - Optionally serialised to disk (faiss.write_index / read_index) for instant reload.
  - IVF-PQ compresses 1536-byte float32 vectors to 96 bytes -> 16x compression.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from schemas import (
    IndexBenchmark,
    IndexType,
    RetrievalMethod,
    RetrievalResponse,
    ScaleProjection,
    SearchResult,
)


# ---------------------------------------------------------------------------
# FAISS Retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    FAISS-backed approximate nearest-neighbour retrieval.

    Supports three index types:
      - IndexType.IVF_FLAT  (inverted file with exact search within clusters)
      - IndexType.HNSW      (hierarchical navigable small world)
      - IndexType.IVF_PQ    (inverted file + product quantization — RECOMMENDED)

    Disk caching:
      Call save_index(path) / load_index(path) to persist the FAISS index.
      Reloading is near-instant (<1 s) vs rebuilding which can take minutes.

    Usage:
        retriever = FAISSRetriever(index_type=IndexType.IVF_PQ)
        retriever.fit(embeddings, doc_ids, doc_titles, doc_topics)
        response  = retriever.search_vector("Q-001", query_text, query_vec)
    """

    def __init__(
        self,
        index_type:   IndexType = IndexType.IVF_PQ,
        n_clusters:   int = 256,      # IVF: number of Voronoi cells (nlist)
        n_probe:      int = 32,       # IVF: cells to visit at query time
        m_pq:         int = 96,       # PQ: number of sub-quantizers (bytes/vec)
        bits_pq:      int = 8,        # PQ: bits per sub-quantizer (256 centroids)
        hnsw_m:       int = 32,       # HNSW: number of neighbours per node
        hnsw_ef:      int = 64,       # HNSW: ef_construction
        dim:          int = 384,      # embedding dimension
    ) -> None:
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.n_probe    = n_probe
        self.m_pq       = m_pq
        self.bits_pq    = bits_pq
        self.hnsw_m     = hnsw_m
        self.hnsw_ef    = hnsw_ef
        self.dim        = dim

        self._index         = None      # faiss.Index
        self._doc_ids:    list[str] = []
        self._doc_titles: list[str] = []
        self._doc_topics: list[str] = []
        self._build_time_s: float   = 0.0
        self._n_docs:       int     = 0
        self._query_times:  list[float] = []

    # ------------------------------------------------------------------
    # Task 7: IMPLEMENTED — _build_ivfflat()
    # ------------------------------------------------------------------

    def _build_ivfflat(self, embeddings: np.ndarray) -> "faiss.Index":
        """
        Build an IVFFlat index.

        IVFFlat partitions the vector space into n_clusters Voronoi cells.
        At query time, only n_probe cells are searched — ~8x speed-up vs flat
        at ~95% recall.

        Note: same memory as flat (no compression), so NOT suitable for 10M docs.
        """
        import faiss

        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(
            quantizer, self.dim, self.n_clusters, faiss.METRIC_INNER_PRODUCT
        )
        # Train: learn cluster centroids from embeddings
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = self.n_probe        # cells searched per query
        return index

    # ------------------------------------------------------------------
    # Task 8: IMPLEMENTED — _build_hnsw()
    # ------------------------------------------------------------------

    def _build_hnsw(self, embeddings: np.ndarray) -> "faiss.Index":
        """
        Build an HNSW (Hierarchical Navigable Small World) index.

        HNSW builds a layered proximity graph — very fast queries (~2-5 ms)
        and ~98% recall@10.

        DOWNSIDE: each node also stores M=32 graph edges x 4 bytes x 2 layers
        ~= +256 bytes/vec on top of the 1536 byte dense vector.
        At 10M docs that's ~18 GB — exceeds 16 GB RAM limit.

        Does NOT require .train() — add directly.
        """
        import faiss

        index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef
        index.add(embeddings)    # No .train() for HNSW
        return index

    # ------------------------------------------------------------------
    # Task 9: IMPLEMENTED — _build_ivfpq()  <-- KEY TASK
    # ------------------------------------------------------------------

    def _build_ivfpq(self, embeddings: np.ndarray) -> "faiss.Index":
        """
        Build an IVF-PQ (Inverted File with Product Quantization) index.

        RECOMMENDED for the 10M-document hospital deployment.

        Product Quantization compresses each 384-dim float32 vector (1536 bytes)
        into m_pq bytes (default 96 bytes) — a 16x compression:

          Original: 384 dims * 4 bytes = 1536 bytes/vector
          PQ step:  Split 384 dims into M=96 sub-vectors of 4 dims each.
                    For each sub-vector, find the nearest centroid from 256 options.
                    Store index (1 byte) instead of 4 floats (16 bytes).
          Result:   96 sub-vectors * 1 byte = 96 bytes/vector

          At 10M docs:
            Dense:   10M * 1536 = 14.4 GB  -> EXCEEDS 16 GB RAM  BAD
            IVF-PQ:  10M *   96 =  0.9 GB  -> fits easily          GOOD

        Recall: ~85-90% vs exact search.  Acceptable for clinical search when
        combined with the multi-stage pipeline (BM25 pre-filter + SBERT re-rank).

        Key constraint: m_pq must evenly divide self.dim (384 / 96 = 4 dims/sub-vec).
        """
        import faiss

        # Validate m_pq divisibility
        assert self.dim % self.m_pq == 0, (
            f"m_pq={self.m_pq} must evenly divide dim={self.dim}. "
            f"Valid options: {[m for m in [8,12,16,24,32,48,64,96] if self.dim % m == 0]}"
        )

        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.n_clusters,   # nlist: number of Voronoi cells
            self.m_pq,         # M: sub-quantizers = bytes per vector
            self.bits_pq,      # nbits: 8 bits -> 256 centroids per sub-quantizer
        )
        # IVF-PQ MUST be trained before adding
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = self.n_probe    # cells searched at query time
        return index

    # ------------------------------------------------------------------
    # fit() — dispatches to correct builder
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings:  np.ndarray,
        doc_ids:     list[str],
        doc_titles:  list[str],
        doc_topics:  list[str],
        cache_path:  Optional[Path] = None,
    ) -> None:
        """
        Build FAISS index from pre-computed L2-normalised embeddings.

        Args:
            embeddings: float32 array of shape (N, dim).
            doc_ids:    list of document IDs (length N).
            doc_titles: list of document titles.
            doc_topics: list of ground-truth topic labels.
            cache_path: if given, save/load the FAISS index from disk.
        """
        import faiss

        assert embeddings.dtype == np.float32, "embeddings must be float32"
        assert embeddings.shape[1] == self.dim, \
            f"embeddings dim {embeddings.shape[1]} != expected {self.dim}"

        self._doc_ids    = doc_ids
        self._doc_titles = doc_titles
        self._doc_topics = doc_topics
        self._n_docs     = len(doc_ids)

        # Try to load from cache first
        if cache_path and Path(cache_path).exists():
            print(f"[FAISS/{self.index_type.value}] Loading index from {cache_path} ...")
            t0 = time.perf_counter()
            self._index = faiss.read_index(str(cache_path))
            self._build_time_s = time.perf_counter() - t0
            print(f"[FAISS/{self.index_type.value}] Loaded in {self._build_time_s:.2f}s")
            return

        t0 = time.perf_counter()

        if self.index_type == IndexType.IVF_FLAT:
            self._index = self._build_ivfflat(embeddings)
        elif self.index_type == IndexType.HNSW:
            self._index = self._build_hnsw(embeddings)
        elif self.index_type == IndexType.IVF_PQ:
            self._index = self._build_ivfpq(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self._build_time_s = time.perf_counter() - t0
        print(f"[FAISS/{self.index_type.value}] Built in {self._build_time_s:.2f}s "
              f"for {self._n_docs:,} docs")

        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(cache_path))
            print(f"[FAISS/{self.index_type.value}] Index cached -> {cache_path}")

    def search_vector(
        self,
        query_id:   str,
        query_text: str,
        query_vec:  np.ndarray,
        top_k:      int = 10,
    ) -> RetrievalResponse:
        """
        Search using a pre-computed query vector.

        Args:
            query_id:   identifier string.
            query_text: raw text (for response object).
            query_vec:  1D float32 array of shape (dim,).
            top_k:      number of results.
        """
        if self._index is None:
            raise RuntimeError("Call fit() before search_vector()")

        q = query_vec.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        distances, indices = self._index.search(q, top_k)
        elapsed = (time.perf_counter() - t0) * 1000
        self._query_times.append(elapsed)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:    # FAISS returns -1 for unfilled slots
                continue
            results.append(SearchResult(
                rank=rank,
                doc_id=self._doc_ids[idx],
                score=float(dist),
                topic=self._doc_topics[idx],
                title=self._doc_titles[idx],
                retrieval_method=RetrievalMethod.SBERT,
            ))

        return RetrievalResponse(
            query_id=query_id,
            original_query=query_text,
            results=results,
            retrieval_method=RetrievalMethod.SBERT,
            latency_ms=elapsed,
            n_results=len(results),
        )

    def benchmark(self, test_vectors: np.ndarray, n_runs: int = 20) -> IndexBenchmark:
        """
        Measure per-query latency and approximate memory usage.

        Args:
            test_vectors: float32 array of shape (N_queries, dim).
            n_runs:       number of queries to average.
        """
        if self._index is None:
            raise RuntimeError("Call fit() first")

        import faiss

        latencies = []
        vecs = test_vectors[:n_runs]
        for vec in vecs:
            t0 = time.perf_counter()
            self._index.search(vec.reshape(1, -1), 10)
            latencies.append((time.perf_counter() - t0) * 1000)

        # Approximate index size via serialization
        try:
            buf = faiss.serialize_index(self._index)
            mem_mb = len(buf) / (1024 ** 2)
        except Exception:
            mem_mb = self._n_docs * self.m_pq / (1024 ** 2)

        return IndexBenchmark(
            index_type=self.index_type,
            n_docs=self._n_docs,
            d_dim=self.dim,
            build_time_s=round(self._build_time_s, 3),
            query_time_ms=round(float(np.mean(latencies)), 3),
            memory_mb=round(mem_mb, 2),
            notes=f"n_probe={self.n_probe}" if self.index_type != IndexType.HNSW else
                  f"hnsw_m={self.hnsw_m}",
        )

    def avg_query_time_ms(self) -> float:
        return float(np.mean(self._query_times)) if self._query_times else 0.0


# ---------------------------------------------------------------------------
# Scale Analyser  (Task 10: IMPLEMENTED — project())
# ---------------------------------------------------------------------------

class ScaleAnalyser:
    """
    Projects memory and latency to 10 million documents and validates against
    the hospital deployment constraints:
      - 16 GB RAM
      - < 50 ms per query on CPU
    """

    # Approximate bytes-per-vector for each index type at 384 dimensions
    BYTES_PER_VECTOR: dict[IndexType, int] = {
        IndexType.FLAT:     384 * 4,            # 1536 bytes (dense float32)
        IndexType.IVF_FLAT: 384 * 4,            # same as flat + tiny IVF overhead
        IndexType.HNSW:     384 * 4 + 32 * 4 * 2,  # ~1792 bytes (graph links)
        IndexType.IVF_PQ:   96,                 # 96 bytes (m_pq=96, 8-bit codes)
    }

    # Approximate milliseconds per query at 10M docs on a 4-core CPU
    MS_PER_QUERY_10M: dict[IndexType, float] = {
        IndexType.FLAT:     250.0,
        IndexType.IVF_FLAT: 30.0,
        IndexType.HNSW:     4.0,
        IndexType.IVF_PQ:   12.0,
    }

    RECOMMENDATION_REASONS: dict[IndexType, str] = {
        IndexType.FLAT:    "Exact but too slow and too much RAM for 10M docs.",
        IndexType.IVF_FLAT:"Meets latency SLA but exceeds 16 GB RAM at 10M docs.",
        IndexType.HNSW:    "Fastest query but highest memory — exceeds 16 GB RAM.",
        IndexType.IVF_PQ:  "Only index satisfying BOTH 16 GB RAM and 50 ms SLA. "
                           "16x compression via Product Quantization, ~85-90% recall — "
                           "acceptable for clinical retrieval when combined with re-ranking.",
    }

    def __init__(self, n_docs: int = 10_000_000) -> None:
        self.n_docs = n_docs

    # ------------------------------------------------------------------
    # Task 10: IMPLEMENTED — project()
    # ------------------------------------------------------------------

    def project(self) -> list[ScaleProjection]:
        """
        For each index type, compute projected memory and latency at self.n_docs,
        then return Pydantic-validated ScaleProjection objects.

        Algorithm for each index_type:
          bytes_total  = self.n_docs * self.BYTES_PER_VECTOR[index_type]
          memory_gb    = bytes_total / (1024**3)
          latency_ms   = self.MS_PER_QUERY_10M[index_type]
          fits_ram     = memory_gb <= 16.0
          meets_sla    = latency_ms <= 50.0
          recommended  = fits_ram AND meets_sla

        Returns:
            list[ScaleProjection] sorted by projected_memory_gb ascending.
        """
        flat_bytes = self.BYTES_PER_VECTOR[IndexType.FLAT]
        projections = []

        for index_type in IndexType:
            if index_type == IndexType.FLAT:
                # Flat is a special case — exact search, reference baseline
                bytes_per_vec = flat_bytes
                latency_ms    = self.MS_PER_QUERY_10M[IndexType.FLAT]
            else:
                bytes_per_vec = self.BYTES_PER_VECTOR[index_type]
                latency_ms    = self.MS_PER_QUERY_10M[index_type]

            bytes_total = self.n_docs * bytes_per_vec
            memory_gb   = bytes_total / (1024 ** 3)
            fits_ram    = memory_gb <= 16.0
            meets_sla   = latency_ms <= 50.0
            recommended = fits_ram and meets_sla

            compression_ratio = flat_bytes / bytes_per_vec if bytes_per_vec > 0 else None

            proj = ScaleProjection(
                index_type=index_type,
                n_docs_projected=self.n_docs,
                projected_memory_gb=round(memory_gb, 2),
                projected_latency_ms=round(latency_ms, 1),
                fits_in_16gb_ram=fits_ram,
                meets_50ms_sla=meets_sla,
                compression_ratio=round(compression_ratio, 1) if compression_ratio else None,
                recommended=recommended,
                recommendation_reason=self.RECOMMENDATION_REASONS[index_type],
            )
            projections.append(proj)

        # Sort by projected memory ascending
        projections.sort(key=lambda p: p.projected_memory_gb)
        return projections

    def print_table(self, projections: list[ScaleProjection]) -> None:
        """Pretty-print the scale projection table."""
        print(f"\n{'='*80}")
        print(f"  Scale Projection: {self.n_docs:,} documents")
        print(f"  Constraints: RAM <= 16 GB, Latency <= 50 ms on CPU")
        print(f"{'='*80}")
        print(f"  {'Index':<12} {'Memory':>10} {'Latency':>10} {'RAM OK':>8} "
              f"{'SLA OK':>8} {'Compress':>10} {'Rec':>5}")
        print("  " + "-" * 68)
        for p in projections:
            compress = f"{p.compression_ratio:.0f}x" if p.compression_ratio else "1x"
            print(
                f"  {p.index_type.value:<12} "
                f"{p.projected_memory_gb:>8.1f}GB "
                f"{p.projected_latency_ms:>9.0f}ms "
                f"{'YES' if p.fits_in_16gb_ram else 'NO':>8} "
                f"{'YES' if p.meets_50ms_sla else 'NO':>8} "
                f"{compress:>10} "
                f"{'***' if p.recommended else '':>5}"
            )
        print()
        recommended = [p for p in projections if p.recommended]
        if recommended:
            rec = recommended[0]
            print(f"  RECOMMENDATION: {rec.index_type.value.upper()}")
            print(f"  {rec.recommendation_reason}")
        print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Scale Projection (Task 10) ===")
    analyser = ScaleAnalyser()
    projections = analyser.project()
    analyser.print_table(projections)

    print("\n=== Quick FAISS build demo (needs SBERT embeddings) ===")
    print("Run main.py --part D for full FAISS benchmarks.")
