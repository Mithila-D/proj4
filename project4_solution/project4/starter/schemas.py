"""
Pydantic Output Schemas — Project 4: Semantic Search & Retrieval
================================================================
DO NOT MODIFY this file.

All pipeline outputs must be validated against these schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Enumerations
# ============================================================

class RetrievalMethod(str, Enum):
    TFIDF        = "tfidf"
    BM25         = "bm25"
    SBERT        = "sbert"
    EXPANDED_BM25 = "expanded_bm25"
    MULTISTAGE   = "multistage"


class IndexType(str, Enum):
    FLAT     = "flat"       # brute-force exact search (baseline)
    IVF_FLAT = "ivfflat"    # inverted file index, exact within cells
    HNSW     = "hnsw"       # hierarchical navigable small world graph
    IVF_PQ   = "ivfpq"      # inverted file + product quantization (recommended)


# ============================================================
# Query expansion schema
# ============================================================

class QueryExpansion(BaseModel):
    """
    Result of expanding a user query with medical synonyms.

    Example:
        original_query  = "heart attack treatment aspirin antiplatelet"
        expanded_terms  = ["myocardial infarction", "STEMI", "NSTEMI",
                           "acute coronary syndrome", "antiplatelet therapy",
                           "clopidogrel", "ticagrelor"]
        expansion_query = "heart attack myocardial infarction STEMI NSTEMI ..."
        n_terms_added   = 7
        ontology_hits   = {"heart attack": ["myocardial infarction", "STEMI", ...]}
    """
    original_query:   str
    expanded_terms:   list[str]          = Field(default_factory=list)
    expansion_query:  str                # original + expanded terms concatenated
    n_terms_added:    int
    ontology_hits:    dict[str, list[str]]   # which source terms triggered expansion

    @field_validator("n_terms_added")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("n_terms_added must be >= 0")
        return v

    @model_validator(mode="after")
    def expansion_contains_original(self) -> "QueryExpansion":
        if self.original_query.lower() not in self.expansion_query.lower():
            raise ValueError("expansion_query must contain the original_query text")
        return self


# ============================================================
# Individual search result
# ============================================================

class SearchResult(BaseModel):
    """
    A single retrieved document with its score.

    rank           : 1-indexed rank in the result list
    doc_id         : unique document identifier
    score          : similarity/relevance score (higher = more relevant)
    topic          : ground-truth topic label (for evaluation)
    title          : document title (snippet)
    retrieval_method: which pipeline stage produced this result
    is_relevant    : whether this doc truly answers the query (for NDCG / MRR)
    """
    rank:             int
    doc_id:           str
    score:            float
    topic:            Optional[str]  = None
    title:            Optional[str]  = None
    retrieval_method: RetrievalMethod
    is_relevant:      Optional[bool] = None     # None = unknown (not evaluated)

    @field_validator("rank")
    @classmethod
    def rank_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rank must be >= 1")
        return v


# ============================================================
# Full retrieval response
# ============================================================

class RetrievalResponse(BaseModel):
    """
    Complete response for one query, including expansion details and ranked results.
    """
    query_id:          str
    original_query:    str
    expansion:         Optional[QueryExpansion]  = None   # None for baselines
    results:           list[SearchResult]        = Field(default_factory=list)
    retrieval_method:  RetrievalMethod
    latency_ms:        Optional[float]           = None
    n_results:         int

    @model_validator(mode="after")
    def n_results_consistent(self) -> "RetrievalResponse":
        if self.n_results != len(self.results):
            raise ValueError(
                f"n_results ({self.n_results}) != len(results) ({len(self.results)})"
            )
        return self

    @model_validator(mode="after")
    def ranks_sequential(self) -> "RetrievalResponse":
        for i, r in enumerate(self.results, start=1):
            if r.rank != i:
                raise ValueError(
                    f"results[{i-1}].rank is {r.rank}, expected {i} (1-indexed sequential)"
                )
        return self


# ============================================================
# Index benchmark — one row per index type
# ============================================================

class IndexBenchmark(BaseModel):
    """
    Benchmark results for a single FAISS index configuration.

    Used in Task 9 to compare IVFFlat / HNSW / IVF-PQ.
    """
    index_type:       IndexType
    n_docs:           int
    d_dim:            int            = 384       # SBERT embedding dimension
    build_time_s:     float
    query_time_ms:    float          # per-query average
    memory_mb:        float
    recall_at_10:     Optional[float] = None     # recall@10 vs brute-force ground truth
    notes:            Optional[str]   = None

    @field_validator("query_time_ms", "build_time_s", "memory_mb")
    @classmethod
    def positive_times(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Timing and memory values must be non-negative")
        return v


# ============================================================
# Scale projection — memory / latency at 10 M documents
# ============================================================

class ScaleProjection(BaseModel):
    """
    Projected memory and latency at 10 million documents.
    Used in Task 10 for the RAM-constrained hospital deployment analysis.

    Constraints:
      - RAM budget  : 16 GB
      - Latency SLA : 50 ms per query on CPU
    """
    index_type:            IndexType
    n_docs_projected:      int               = 10_000_000
    projected_memory_gb:   float
    projected_latency_ms:  float
    fits_in_16gb_ram:      bool
    meets_50ms_sla:        bool
    compression_ratio:     Optional[float]  = None    # relative to flat index
    recommended:           bool
    recommendation_reason: Optional[str]   = None

    @model_validator(mode="after")
    def consistency_check(self) -> "ScaleProjection":
        fits  = self.projected_memory_gb <= 16.0
        fast  = self.projected_latency_ms <= 50.0
        if self.fits_in_16gb_ram != fits:
            raise ValueError(
                f"fits_in_16gb_ram={self.fits_in_16gb_ram} but "
                f"projected_memory_gb={self.projected_memory_gb:.1f} (threshold 16 GB)"
            )
        if self.meets_50ms_sla != fast:
            raise ValueError(
                f"meets_50ms_sla={self.meets_50ms_sla} but "
                f"projected_latency_ms={self.projected_latency_ms:.1f} (threshold 50 ms)"
            )
        return self
