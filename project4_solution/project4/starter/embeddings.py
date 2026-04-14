"""
Dense Retrieval — SBERT Embeddings + Cosine Similarity  (Tasks 5 & 6 IMPLEMENTED)
==================================================================================
Transforms texts to 384-dimensional dense vectors using sentence-transformers.
Similarity is measured by cosine distance in embedding space.

Unlike TF-IDF / BM25, SBERT embeddings capture SEMANTIC meaning:
  "heart attack" -> embedding close to "myocardial infarction"
  "blood thinner" -> embedding close to "anticoagulant"

Speed-up strategy for subsequent queries:
  - Document embeddings are encoded ONCE and stored in RAM as a numpy matrix.
  - Optionally saved to a .npy file so that subsequent runs skip encoding.
  - Each query: encode one short string (~2-10 ms) + matrix dot product (~1 ms).
  - Total subsequent-query latency for 2000 docs: ~3-12 ms on CPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from schemas import RetrievalMethod, RetrievalResponse, SearchResult


# ---------------------------------------------------------------------------
# Embedding Encoder
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """
    Sentence-level encoder using sentence-transformers.

    Default model: 'all-MiniLM-L6-v2'  (384-dim, fast, strong baseline)
    Medical model: 'pritamdeka/S-PubMedBert-MS-MARCO'  (better for clinical text)

    Memory:
      2000 documents x 384 dims x 4 bytes (float32) = ~3 MB  (trivial)
      10M  documents x 384 dims x 4 bytes (float32) = ~14.4 GB  (exceeds 16 GB RAM!)
      -> For 10M docs, use IVF-PQ compression (see faiss_index.py)
    """

    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device:     str = "cpu",
        cache_dir:  Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device     = device
        self.cache_dir  = cache_dir
        self._model     = None

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model (only on first use)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[SBERT] Loading model: {self.model_name} on {self.device} ...")
            kwargs = {}
            if self.cache_dir:
                kwargs["cache_folder"] = str(self.cache_dir)
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[SBERT] Model loaded in {elapsed:.1f}s — "
                  f"dim={self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    # ------------------------------------------------------------------
    # Task 5: IMPLEMENTED — encode()
    # ------------------------------------------------------------------

    def encode(
        self,
        texts:           list[str],
        show_progress:   bool = True,
        normalize:       bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised dense vectors.

        Args:
            texts:         list of strings to encode.
            show_progress: show tqdm progress bar during batch encoding.
            normalize:     if True, L2-normalise each vector so that
                           cosine similarity == dot product.

        Returns:
            np.ndarray of shape (len(texts), 384), dtype float32.

        Speed note:
            For 2000 docs on CPU, encoding takes ~30-90 s the first time.
            Subsequent queries re-use the cached matrix — only ONE new
            text is encoded per query (~2-10 ms).
        """
        self._load_model()

        t0 = time.perf_counter()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0

        if show_progress and len(texts) > 1:
            print(f"[SBERT] Encoded {len(texts)} texts in {elapsed:.1f}s "
                  f"({len(texts)/elapsed:.0f} texts/s)")

        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Convenience wrapper for a single text string. Very fast (~2-10 ms)."""
        return self.encode([text], show_progress=False, normalize=normalize)[0]

    @property
    def dim(self) -> int:
        return self.EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Dense Retriever (brute-force exact cosine search)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Exact cosine similarity retrieval over a precomputed embedding matrix.

    For corpora up to ~50 000 documents this is fast enough on CPU.
    For larger corpora, use FAISSRetriever in faiss_index.py.

    Speed-up:
      - Document embeddings encoded ONCE in fit().
      - Saved to .npy cache so that next run loads in ~0.1 s instead of ~60 s.
      - Per-query cost: encode_single() + matrix @ vec — typically 3-15 ms.
    """

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self.encoder      = encoder
        self._doc_matrix: Optional[np.ndarray] = None    # (N, D) float32
        self._doc_ids:    list[str]             = []
        self._doc_titles: list[str]             = []
        self._doc_topics: list[str]             = []
        self._fitted      = False
        self._query_times: list[float]          = []

    def fit(self, docs: pd.DataFrame, cache_path: Optional[Path] = None) -> None:
        """
        Encode all documents and store the embedding matrix.

        Cache strategy:
          If cache_path exists, load the pre-computed matrix (fast).
          Otherwise encode all docs and save to cache_path.

        Args:
            docs:        DataFrame with [doc_id, full_text, title, topic].
            cache_path:  path to a .npy file for embedding cache.
        """
        self._doc_ids    = docs["doc_id"].tolist()
        self._doc_titles = docs["title"].tolist()
        self._doc_topics = docs["topic"].tolist() if "topic" in docs.columns else [""] * len(docs)

        if cache_path and Path(cache_path).exists():
            print(f"[DenseRetriever] Loading cached embeddings from {cache_path} ...")
            t0 = time.perf_counter()
            self._doc_matrix = np.load(str(cache_path)).astype(np.float32)
            elapsed = time.perf_counter() - t0
            print(f"[DenseRetriever] Loaded {self._doc_matrix.shape} in {elapsed:.2f}s")
        else:
            print(f"[DenseRetriever] Encoding {len(docs)} documents ...")
            texts = docs["full_text"].fillna("").tolist()
            self._doc_matrix = self.encoder.encode(texts, show_progress=True)
            if cache_path:
                cache_path = Path(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), self._doc_matrix)
                print(f"[DenseRetriever] Embeddings cached -> {cache_path}")

        print(f"[DenseRetriever] Dense index ready: {self._doc_matrix.shape} "
              f"({self.memory_mb:.1f} MB)")
        self._fitted = True

    # ------------------------------------------------------------------
    # Task 6: IMPLEMENTED — search()
    # ------------------------------------------------------------------

    def search(
        self,
        query_id:  str,
        query:     str,
        top_k:     int = 10,
    ) -> RetrievalResponse:
        """
        Retrieve top-k documents by cosine similarity to the query embedding.

        Since embeddings are L2-normalised, cosine_sim == dot_product:
            scores = self._doc_matrix @ q_vec   (shape: N,)

        Subsequent queries reuse the stored matrix — only the query string
        needs to be encoded each time (~2-10 ms on CPU).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before search()")

        t0 = time.perf_counter()

        # Encode query (single string, fast)
        q_vec  = self.encoder.encode_single(query, normalize=True)   # shape: (D,)

        # Cosine similarity via dot product (vectors are L2-normalised)
        scores = self._doc_matrix @ q_vec                             # shape: (N,)

        # Rank descending
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
                retrieval_method=RetrievalMethod.SBERT,
            )
            for rank, i in enumerate(top_idx)
        ]

        return RetrievalResponse(
            query_id=query_id,
            original_query=query,
            results=results,
            retrieval_method=RetrievalMethod.SBERT,
            latency_ms=elapsed,
            n_results=len(results),
        )

    def avg_query_time_ms(self) -> float:
        return float(np.mean(self._query_times)) if self._query_times else 0.0

    @property
    def memory_mb(self) -> float:
        """Approximate memory used by the embedding matrix."""
        if self._doc_matrix is None:
            return 0.0
        return self._doc_matrix.nbytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Memory calculator (Task 10 support)
# ---------------------------------------------------------------------------

def estimate_embedding_memory(
    n_docs:       int,
    embedding_dim: int = 384,
    dtype:         str = "float32",
    compression:   Optional[str] = None,
) -> dict[str, float]:
    """
    Calculate memory required for embedding matrix at different scales.

    Args:
        n_docs:         Number of documents.
        embedding_dim:  Embedding dimensionality (default: 384 for MiniLM).
        dtype:          'float32' (4 bytes) or 'float16' (2 bytes).
        compression:    None | 'pq96'  (IVF-PQ with 96 bytes/vector)

    Returns:
        dict with 'bytes', 'mb', 'gb'
    """
    bytes_per_val = 4 if dtype == "float32" else 2

    if compression == "pq96":
        total_bytes = n_docs * 96
        method      = "IVF-PQ (96 bytes/vec)"
    elif compression == "pq48":
        total_bytes = n_docs * 48
        method      = "IVF-PQ (48 bytes/vec)"
    else:
        total_bytes = n_docs * embedding_dim * bytes_per_val
        method      = f"Dense {dtype}"

    result = {
        "n_docs":  n_docs,
        "method":  method,
        "bytes":   total_bytes,
        "mb":      total_bytes / (1024 ** 2),
        "gb":      total_bytes / (1024 ** 3),
    }
    print(f"  {method}: {n_docs:>10,} docs -> {result['gb']:6.2f} GB")
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Memory scale analysis ===")
    for n in [2_000, 100_000, 1_000_000, 10_000_000]:
        estimate_embedding_memory(n)
    print()
    for n in [1_000_000, 10_000_000]:
        estimate_embedding_memory(n, compression="pq96")

    print("\n=== Semantic similarity demo ===")
    encoder = EmbeddingEncoder()
    queries = [
        "heart attack treatment aspirin",
        "myocardial infarction antiplatelet therapy",
        "blood poisoning ICU antibiotics",
        "sepsis management vasopressors",
    ]
    vecs = encoder.encode(queries, show_progress=False)
    print(f"Encoded {len(queries)} queries -> shape {vecs.shape}")
    print("\nPairwise cosine similarities (should be high for same concept):")
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            sim = float(np.dot(vecs[i], vecs[j]))
            print(f"  sim({i},{j}) = {sim:.3f}")
            print(f"    '{queries[i][:45]}'")
            print(f"    '{queries[j][:45]}'")
