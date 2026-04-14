"""
Information Retrieval Evaluation Utilities
==========================================
Standard IR metrics:
  - MRR@K   (Mean Reciprocal Rank)
  - NDCG@K  (Normalised Discounted Cumulative Gain)
  - Recall@K
  - Precision@K
  - Latency measurement and SLA compliance

These are the metrics used in real production retrieval systems.
Unlike classification metrics (F1, accuracy), IR metrics reward:
  - Having a relevant result at rank 1 (MRR)
  - Having relevant results ranked higher (NDCG)
  - Covering all relevant documents (Recall)

Clinical context: for a life-critical drug-interaction query, missing the
single most relevant guideline (FN at rank 1) is far more dangerous than
returning an irrelevant result at rank 10.  MRR captures this penalty.
"""

from __future__ import annotations

import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from schemas import IndexBenchmark, RetrievalResponse, ScaleProjection


# ============================================================
# Core IR metrics
# ============================================================

def mrr_at_k(relevance: list[int], k: int) -> float:
    """
    Mean Reciprocal Rank at K.

    MRR = 1 / rank_of_first_relevant_document

    If no relevant document in top-K, return 0.

    Args:
        relevance: binary list, 1 = relevant, 0 = not relevant (length K).
        k:         cutoff rank.

    Returns:
        float in [0, 1]
    """
    for rank, rel in enumerate(relevance[:k], start=1):
        if rel == 1:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevance: list[int], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    return sum(relevance[:k]) / k if k > 0 else 0.0


def recall_at_k(relevance: list[int], k: int, n_relevant: Optional[int] = None) -> float:
    """
    Fraction of all relevant documents found in top-K.

    If n_relevant is None, uses sum(relevance) as the total number of relevant docs.
    """
    total = n_relevant if n_relevant is not None else sum(relevance)
    if total == 0:
        return 0.0
    return sum(relevance[:k]) / total


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    DCG@K  = sum_i rel_i / log2(i+2)
    IDCG@K = DCG of ideal ranking (all relevant docs ranked first)
    NDCG@K = DCG@K / IDCG@K

    NDCG penalises relevant documents ranked lower more than MRR.

    Args:
        relevance: binary or graded relevance list.
        k:         cutoff.

    Returns:
        float in [0, 1]
    """
    dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[:k]))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Batch evaluation over query set
# ============================================================

def evaluate_retrieval(
    responses:   list[RetrievalResponse],
    queries_df:  pd.DataFrame,          # must have columns: query_id, relevant_topics
    k:           int = 10,
    model_name:  str = "system",
    verbose:     bool = True,
) -> dict:
    """
    Evaluate a list of retrieval responses against ground-truth relevance.

    Ground truth: result is relevant if its topic appears in the query's
    relevant_topics field (semicolon-separated).

    Args:
        responses:   list of RetrievalResponse, one per query.
        queries_df:  query test set with relevant_topics column.
        k:           evaluation cutoff.
        model_name:  label for printing.
        verbose:     print results table.

    Returns:
        dict with MRR@K, Precision@K, Recall@K, NDCG@K (all averaged).
    """
    id_to_topics = {
        row["query_id"]: set(row["relevant_topics"].split("; "))
        for _, row in queries_df.iterrows()
    }

    mrr_scores    = []
    prec_scores   = []
    rec_scores    = []
    ndcg_scores   = []
    latency_ms    = []

    per_query = []

    for resp in responses:
        relevant_topics = id_to_topics.get(resp.query_id, set())
        relevance = [
            1 if (r.topic in relevant_topics) else 0
            for r in resp.results
        ]

        m   = mrr_at_k(relevance, k)
        p   = precision_at_k(relevance, k)
        r   = recall_at_k(relevance, k)
        n   = ndcg_at_k(relevance, k)

        mrr_scores.append(m)
        prec_scores.append(p)
        rec_scores.append(r)
        ndcg_scores.append(n)
        if resp.latency_ms is not None:
            latency_ms.append(resp.latency_ms)

        per_query.append({
            "query_id":  resp.query_id,
            "query":     resp.original_query[:50],
            "mrr":       round(m, 3),
            "ndcg":      round(n, 3),
            "recall":    round(r, 3),
        })

    avg_mrr    = float(np.mean(mrr_scores))
    avg_prec   = float(np.mean(prec_scores))
    avg_rec    = float(np.mean(rec_scores))
    avg_ndcg   = float(np.mean(ndcg_scores))
    avg_latency = float(np.mean(latency_ms)) if latency_ms else None

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Evaluation: {model_name}  (K={k})")
        print(f"{'='*55}")
        print(f"  MRR@{k}         : {avg_mrr:.4f}")
        print(f"  Precision@{k}   : {avg_prec:.4f}")
        print(f"  Recall@{k}      : {avg_rec:.4f}")
        print(f"  NDCG@{k}        : {avg_ndcg:.4f}")
        if avg_latency is not None:
            print(f"  Avg latency    : {avg_latency:.1f} ms")

        # Bottom 5 queries by MRR (where the system struggles)
        worst = sorted(per_query, key=lambda x: x["mrr"])[:5]
        print(f"\n  5 hardest queries (lowest MRR):")
        for q in worst:
            print(f"    [{q['query_id']}] MRR={q['mrr']:.3f}  {q['query']}")

    return dict(
        model=model_name,
        mrr=avg_mrr,
        precision=avg_prec,
        recall=avg_rec,
        ndcg=avg_ndcg,
        avg_latency_ms=avg_latency,
        per_query=per_query,
    )


# ============================================================
# Benchmark comparison table
# ============================================================

def print_comparison_table(
    eval_results: list[dict],
    k:            int = 10,
) -> None:
    """
    Print side-by-side IR metrics for multiple retrieval models.

    Args:
        eval_results: list of dicts from evaluate_retrieval().
        k:            cutoff (for column header).
    """
    print(f"\n{'='*70}")
    print(f"  Retrieval Method Comparison  (K={k})")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'MRR@K':>8} {'Prec@K':>8} {'Rec@K':>8} {'NDCG@K':>8} {'ms':>8}")
    print("  " + "-" * 66)
    for r in eval_results:
        lat = f"{r['avg_latency_ms']:.1f}" if r.get("avg_latency_ms") else "-"
        print(
            f"  {r['model']:<30} "
            f"{r['mrr']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['ndcg']:>8.4f} "
            f"{lat:>8}"
        )
    print()
    if len(eval_results) >= 2:
        best = max(eval_results, key=lambda x: x["mrr"])
        worst = min(eval_results, key=lambda x: x["mrr"])
        improvement = (best["mrr"] - worst["mrr"]) / max(worst["mrr"], 1e-9) * 100
        print(f"  Best:  {best['model']} (MRR={best['mrr']:.4f})")
        print(f"  Worst: {worst['model']} (MRR={worst['mrr']:.4f})")
        print(f"  Relative improvement: +{improvement:.1f}%")


# ============================================================
# FAISS benchmark + scale visualisation
# ============================================================

def plot_index_benchmarks(benchmarks: list[IndexBenchmark]) -> None:
    """
    Side-by-side bar charts of query latency and memory for each index type.
    """
    names     = [b.index_type.value for b in benchmarks]
    latencies = [b.query_time_ms for b in benchmarks]
    memories  = [b.memory_mb for b in benchmarks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(names, latencies, color="steelblue")
    axes[0].axhline(50, color="red", linestyle="--", label="50 ms SLA")
    axes[0].set_title("Query Latency (ms)")
    axes[0].set_ylabel("ms")
    axes[0].legend()

    axes[1].bar(names, memories, color="coral")
    axes[1].set_title("Index Memory (MB)")
    axes[1].set_ylabel("MB")

    plt.tight_layout()
    plt.savefig("outputs/index_benchmarks.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/index_benchmarks.png")


def plot_scale_projection(projections: list[ScaleProjection]) -> None:
    """
    Visualise projected memory and latency at 10M docs for each index type.
    """
    names    = [p.index_type.value for p in projections]
    memories = [p.projected_memory_gb for p in projections]
    latencies = [p.projected_latency_ms for p in projections]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colours = ["red" if not p.fits_in_16gb_ram else "green" for p in projections]
    axes[0].bar(names, memories, color=colours)
    axes[0].axhline(16, color="orange", linestyle="--", label="16 GB RAM limit")
    axes[0].set_title("Projected Memory @ 10M docs (GB)")
    axes[0].set_ylabel("GB")
    axes[0].legend()

    colours2 = ["red" if not p.meets_50ms_sla else "green" for p in projections]
    axes[1].bar(names, latencies, color=colours2)
    axes[1].axhline(50, color="orange", linestyle="--", label="50 ms SLA")
    axes[1].set_title("Projected Latency @ 10M docs (ms)")
    axes[1].set_ylabel("ms")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/scale_projection.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/scale_projection.png")


def plot_mrr_by_query_type(
    responses_by_method: dict[str, list[RetrievalResponse]],
    queries_df:          pd.DataFrame,
    k:                   int = 10,
) -> None:
    """
    Heatmap of MRR per query × retrieval method.
    Useful for identifying which query types each method handles best.
    """
    id_to_topics = {
        row["query_id"]: set(row["relevant_topics"].split("; "))
        for _, row in queries_df.iterrows()
    }
    query_ids = queries_df["query_id"].tolist()
    methods   = list(responses_by_method.keys())

    matrix = np.zeros((len(query_ids), len(methods)))

    for j, method in enumerate(methods):
        for i, (qid, resp) in enumerate(zip(query_ids, responses_by_method[method])):
            relevant_topics = id_to_topics.get(qid, set())
            relevance = [1 if (r.topic in relevant_topics) else 0 for r in resp.results]
            matrix[i, j] = mrr_at_k(relevance, k)

    plt.figure(figsize=(max(8, len(methods) * 2), max(10, len(query_ids) * 0.3)))
    sns.heatmap(
        matrix,
        xticklabels=methods,
        yticklabels=[f"{r['query_id']}: {r['query_text'][:30]}" for _, r in queries_df.iterrows()],
        cmap="RdYlGn",
        vmin=0, vmax=1,
        annot=len(query_ids) <= 30,
        fmt=".2f",
    )
    plt.title(f"MRR@{k} per query per method")
    plt.tight_layout()
    plt.savefig("outputs/mrr_heatmap.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/mrr_heatmap.png")
