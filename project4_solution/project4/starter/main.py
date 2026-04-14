"""
Main Entry Point — Project 4: Semantic Search & Retrieval at Scale
===================================================================
Runs all 7 retrieval methods end-to-end with timing, accuracy metrics,
and comparative graphs.

Methods:
  1. TF-IDF                 (sparse, lexical)
  2. BM25                   (sparse, lexical, better TF saturation)
  3. Query Expansion        (ontology-based lay->clinical term mapping)
  4. BM25 + Query Expansion (BM25 with expanded query)
  5. SBERT Dense            (semantic, 384-dim sentence embeddings)
  6. Multi-Stage Pipeline   (BM25 candidates -> SBERT re-rank)
  7. IVF-PQ FAISS           (ANN index, production-scale)

Speed-up strategy:
  - Sparse indices (TF-IDF matrix, BM25) stay in RAM after first fit().
  - SBERT document embeddings saved to cache/corpus_embeddings.npy
  - FAISS indices saved to cache/faiss_<type>.index
  - BM25 fitted objects saved to cache/bm25.pkl
  - TF-IDF fitted objects saved to cache/tfidf.pkl
  - Subsequent runs (or subsequent queries) are MUCH faster.

Run:
    python main.py --part A          # sparse baselines
    python main.py --part B          # SBERT dense
    python main.py --part C          # multi-stage pipeline + ablation
    python main.py --part D          # FAISS benchmarks + scale projection
    python main.py --part all        # everything
    python main.py --part all --k 10 # top-K evaluation cutoff
    python main.py --part all --cache_embeddings   # enable disk caching
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from embeddings import DenseRetriever, EmbeddingEncoder, estimate_embedding_memory
from evaluate import (
    evaluate_retrieval,
    plot_index_benchmarks,
    plot_mrr_by_query_type,
    plot_scale_projection,
    print_comparison_table,
)
from faiss_index import FAISSRetriever, ScaleAnalyser
from pipeline import MultiStagePipeline, run_ablation
from query_expansion import QueryExpander
from retrieval import BM25Retriever, TFIDFRetriever, analyse_vocabulary_mismatch
from schemas import IndexType, RetrievalMethod


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Project 4: Semantic Search & Retrieval")
    p.add_argument("--part",     choices=["A", "B", "C", "D", "all"], default="all",
                   help="Which part to run (A=sparse, B=dense, C=pipeline, D=FAISS, all=everything)")
    p.add_argument("--data_dir", default="data",  help="Path to data directory")
    p.add_argument("--out",      default="outputs", help="Output directory for results/plots")
    p.add_argument("--cache",    default="cache",   help="Directory for cached indices/embeddings")
    p.add_argument("--k",        type=int, default=10, help="Evaluation cutoff K")
    p.add_argument("--device",   default="cpu",     help="Device for SBERT (cpu/cuda)")
    p.add_argument("--model",    default="all-MiniLM-L6-v2", help="SBERT model name")
    p.add_argument("--cache_embeddings", action="store_true", default=True,
                   help="Cache SBERT embeddings and FAISS indices to disk for fast reuse (enabled by default)")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable caching of SBERT embeddings and FAISS indices")
    p.add_argument("--no_plots", action="store_true", help="Skip saving plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility: plot comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    eval_results: list[dict],
    out_dir: Path,
    title: str = "Retrieval Method Comparison",
    filename: str = "metrics_comparison.png",
) -> None:
    """Bar chart of MRR, Recall, NDCG, Precision for all methods."""
    methods  = [r["model"] for r in eval_results]
    metrics  = ["mrr", "recall", "ndcg", "precision"]
    labels   = ["MRR@K", "Recall@K", "NDCG@K", "Precision@K"]
    colors   = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2), 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [r.get(metric, 0) for r in eval_results]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out_path}")


def plot_latency_comparison(
    eval_results: list[dict],
    out_dir: Path,
    filename: str = "latency_comparison.png",
) -> None:
    """Bar chart of per-query latency for all methods."""
    methods   = [r["model"] for r in eval_results if r.get("avg_latency_ms")]
    latencies = [r["avg_latency_ms"] for r in eval_results if r.get("avg_latency_ms")]

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 4))
    bars = ax.bar(methods, latencies, color="steelblue", alpha=0.85)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=9)

    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50 ms SLA")
    ax.set_ylabel("Avg latency per query (ms)")
    ax.set_title("Per-Query Latency by Retrieval Method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out_path}")


def plot_ablation(ablation_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart for ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conditions = ablation_df["condition"].tolist()
    x = np.arange(len(conditions))

    # MRR
    axes[0].bar(x, ablation_df["mrr_at_k"], color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
    for xi, val in zip(x, ablation_df["mrr_at_k"]):
        axes[0].text(xi, val + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.split(".")[0] + ". " + c.split(". ")[1][:15]
                              for c in conditions], rotation=12, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("MRR@K")
    axes[0].set_title("Mean Reciprocal Rank (higher is better)")
    axes[0].grid(axis="y", alpha=0.3)

    # Recall
    axes[1].bar(x, ablation_df["recall_at_k"], color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
    for xi, val in zip(x, ablation_df["recall_at_k"]):
        axes[1].text(xi, val + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.split(".")[0] + ". " + c.split(". ")[1][:15]
                              for c in conditions], rotation=12, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Recall@K")
    axes[1].set_title("Recall@K (higher is better)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Ablation Study: A (BM25) → B (BM25+Exp) → C (SBERT) → D (MultiStage)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "ablation_study.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out_path}")


def plot_mismatch_analysis(mismatch_df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of vocabulary mismatch percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(mismatch_df["overlap_pct"], bins=15, color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Query token overlap with corpus (%)")
    axes[0].set_ylabel("Number of queries")
    axes[0].set_title("Query-Corpus Token Overlap Distribution")
    avg = mismatch_df["overlap_pct"].mean()
    axes[0].axvline(avg, color="red", linestyle="--", label=f"Mean = {avg:.1f}%")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(mismatch_df["lay_mismatch_pct"], bins=10, color="#e74c3c", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Lay-term mismatch (%)")
    axes[1].set_ylabel("Number of queries")
    axes[1].set_title("Lay-Term vs. Clinical Corpus Mismatch")
    n_full = (mismatch_df["lay_mismatch_pct"] == 100).sum()
    axes[1].set_title(
        f"Lay-Term Mismatch  ({n_full}/{len(mismatch_df)} queries: 100% mismatch)"
    )
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "vocabulary_mismatch.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out_path}")


# ---------------------------------------------------------------------------
# Part A — Sparse baselines + vocabulary mismatch
# ---------------------------------------------------------------------------

def run_part_a(
    data_dir:  Path,
    out_dir:   Path,
    cache_dir: Path,
    k:         int,
    use_cache: bool,
    save_plots: bool,
) -> list[dict]:
    print("\n" + "=" * 65)
    print("  PART A — Vocabulary Mismatch + Sparse Baselines")
    print("=" * 65)

    corpus_df = pd.read_csv(data_dir / "clinical_corpus.csv")
    query_df  = pd.read_csv(data_dir / "query_testset.csv")
    print(f"Corpus: {len(corpus_df)} docs | Queries: {len(query_df)}")

    # ---- 1. Vocabulary mismatch analysis ----
    print("\n[Step 1] Vocabulary mismatch analysis")
    mismatch_df = analyse_vocabulary_mismatch(corpus_df, query_df)
    high_mismatch = mismatch_df[mismatch_df["lay_mismatch_pct"] == 100]
    print(f"  Queries with 100% lay-term mismatch: {len(high_mismatch)}/{len(query_df)}")
    print("  These queries will score 0 with TF-IDF / BM25 unless expanded.")
    if save_plots:
        plot_mismatch_analysis(mismatch_df, out_dir)

    # ---- 2. Query expansion preview ----
    print("\n[Step 2] Query expansion preview (Task 2 & 3)")
    expander    = QueryExpander()
    corpus_vocab = expander.build_corpus_vocab(corpus_df["full_text"].fillna("").tolist())
    demo_queries = [
        "heart attack treatment aspirin antiplatelet",
        "water pill swollen ankles fluid retention",
        "brain bleed anticoagulation stop restart",
    ]
    for q in demo_queries:
        exp = expander.expand(q)
        quality = expander.score_expansion_quality(exp, corpus_vocab)
        print(f"  Query: '{q[:55]}'")
        print(f"    Hits: {list(exp.ontology_hits.keys())}  |  +{exp.n_terms_added} terms")
        print(f"    Coverage: {quality['coverage']:.1%}  "
              f"({quality['n_terms_in_corpus']} of {exp.n_terms_added} terms in corpus)")

    # ---- 3. TF-IDF baseline (Method 1) ----
    print("\n[Method 1] TF-IDF baseline")
    tfidf_cache = cache_dir / "tfidf.pkl" if use_cache else None
    tfidf = TFIDFRetriever()
    tfidf.fit(corpus_df, cache_path=tfidf_cache)

    t0 = time.perf_counter()
    responses_tfidf = [
        tfidf.search(row["query_id"], row["query_text"], top_k=k)
        for _, row in query_df.iterrows()
    ]
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Total query time: {total_ms:.0f} ms | "
          f"Avg: {tfidf.avg_query_time_ms():.2f} ms/query")
    metrics_tfidf = evaluate_retrieval(
        responses_tfidf, query_df, k=k, model_name="TF-IDF"
    )
    metrics_tfidf["model"] = "1. TF-IDF"

    # ---- 4. BM25 baseline (Method 2) ----
    print("\n[Method 2] BM25 baseline")
    bm25_cache = cache_dir / "bm25.pkl" if use_cache else None
    bm25 = BM25Retriever()
    bm25.fit(corpus_df, cache_path=bm25_cache)

    t0 = time.perf_counter()
    responses_bm25 = [
        bm25.search(row["query_id"], row["query_text"], top_k=k)
        for _, row in query_df.iterrows()
    ]
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Total query time: {total_ms:.0f} ms | "
          f"Avg: {bm25.avg_query_time_ms():.2f} ms/query")
    metrics_bm25 = evaluate_retrieval(
        responses_bm25, query_df, k=k, model_name="BM25"
    )
    metrics_bm25["model"] = "2. BM25"

    # ---- 5. Query expansion coverage analysis (Method 3) ----
    print("\n[Method 3] Query expansion analysis")
    expander.summary()
    coverage_scores = []
    for _, row in query_df.iterrows():
        exp = expander.expand(row["query_text"])
        q = expander.score_expansion_quality(exp, corpus_vocab)
        coverage_scores.append(q["coverage"])
    print(f"  Avg expansion coverage across queries: {np.mean(coverage_scores):.1%}")

    # ---- 6. BM25 + Query Expansion (Method 4) ----
    print("\n[Method 4] BM25 + Query Expansion")
    bm25_exp = BM25Retriever()
    bm25_exp.fit(corpus_df)

    t0 = time.perf_counter()
    responses_expanded = []
    for _, row in query_df.iterrows():
        exp  = expander.expand(row["query_text"])
        resp = bm25_exp.search(row["query_id"], exp.expansion_query, top_k=k)
        responses_expanded.append(resp)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Total query time: {total_ms:.0f} ms | Avg: {total_ms/len(query_df):.2f} ms/query")
    metrics_expanded = evaluate_retrieval(
        responses_expanded, query_df, k=k, model_name="BM25+Expansion"
    )
    metrics_expanded["model"] = "4. BM25+Expansion"

    # Print comparison and save
    metrics_list = [metrics_tfidf, metrics_bm25, metrics_expanded]
    print_comparison_table(metrics_list, k=k)

    if save_plots:
        plot_metrics_comparison(
            metrics_list, out_dir,
            title=f"Part A — Sparse Baselines (K={k})",
            filename="part_a_sparse_comparison.png"
        )
        plot_latency_comparison(metrics_list, out_dir, filename="part_a_latency.png")

    out_path = out_dir / "part_a_results.json"
    with open(out_path, "w") as f:
        json.dump(
            [{kk: v for kk, v in m.items() if kk != "per_query"} for m in metrics_list],
            f, indent=2,
        )
    print(f"\n  Part A results saved -> {out_path}")

    return metrics_list


# ---------------------------------------------------------------------------
# Part B — Dense retrieval with SBERT (Method 5)
# ---------------------------------------------------------------------------

def run_part_b(
    data_dir:   Path,
    out_dir:    Path,
    cache_dir:  Path,
    k:          int,
    device:     str,
    model:      str,
    use_cache:  bool,
    save_plots: bool,
) -> dict:
    print("\n" + "=" * 65)
    print("  PART B — Dense Retrieval with SBERT  (Method 5)")
    print("=" * 65)

    corpus_df = pd.read_csv(data_dir / "clinical_corpus.csv")
    query_df  = pd.read_csv(data_dir / "query_testset.csv")

    # Memory scale analysis
    print("\n[Memory Scale Analysis]")
    for n in [len(corpus_df), 1_000_000, 10_000_000]:
        estimate_embedding_memory(n)
    estimate_embedding_memory(10_000_000, compression="pq96")

    cache_path = cache_dir / "corpus_embeddings.npy" if use_cache else None

    encoder = EmbeddingEncoder(model_name=model, device=device)
    dense   = DenseRetriever(encoder)

    print(f"\n[Method 5] SBERT Dense Retrieval (model={model})")
    t_fit = time.perf_counter()
    dense.fit(corpus_df, cache_path=cache_path)
    print(f"  Fit time: {(time.perf_counter()-t_fit):.1f}s | "
          f"Embedding matrix: {dense.memory_mb:.1f} MB")

    t0 = time.perf_counter()
    responses_sbert = [
        dense.search(row["query_id"], row["query_text"], top_k=k)
        for _, row in query_df.iterrows()
    ]
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Total query time: {total_ms:.0f} ms | "
          f"Avg: {dense.avg_query_time_ms():.2f} ms/query "
          f"(matrix stays in RAM — all queries same speed)")
    metrics_sbert = evaluate_retrieval(
        responses_sbert, query_df, k=k, model_name="SBERT dense"
    )
    metrics_sbert["model"] = "5. SBERT Dense"

    # Semantic similarity demo
    print("\n[Semantic Match Demo]")
    pairs = [
        ("heart attack treatment aspirin antiplatelet",
         "myocardial infarction antiplatelet therapy clopidogrel"),
        ("blood poisoning ICU antibiotics",
         "sepsis management vasopressors antimicrobial"),
    ]
    for q1, q2 in pairs:
        v1 = encoder.encode_single(q1)
        v2 = encoder.encode_single(q2)
        sim = float(np.dot(v1, v2))
        print(f"  cos('{q1[:40]}',")
        print(f"       '{q2[:40]}') = {sim:.3f}")

    print("\n[Top-3 for 'heart attack treatment aspirin antiplatelet']:")
    resp = dense.search("demo", "heart attack treatment aspirin antiplatelet", top_k=3)
    for r in resp.results:
        print(f"  Rank {r.rank} [{r.topic}] score={r.score:.3f}  {r.title[:60]}")

    if save_plots:
        plot_metrics_comparison(
            [metrics_sbert], out_dir,
            title=f"Part B — SBERT Dense (K={k})",
            filename="part_b_sbert.png"
        )

    return metrics_sbert


# ---------------------------------------------------------------------------
# Part C — Multi-stage pipeline + ablation (Method 6)
# ---------------------------------------------------------------------------

def run_part_c(
    data_dir:   Path,
    out_dir:    Path,
    cache_dir:  Path,
    k:          int,
    device:     str,
    save_plots: bool,
    use_cache:  bool,
) -> None:
    print("\n" + "=" * 65)
    print("  PART C — Multi-Stage Pipeline + Ablation  (Method 6)")
    print("=" * 65)

    corpus_df = pd.read_csv(data_dir / "clinical_corpus.csv")
    query_df  = pd.read_csv(data_dir / "query_testset.csv")

    cache_dir_str = str(cache_dir) if use_cache else None

    t0 = time.perf_counter()
    ablation_df = run_ablation(
        corpus_df, query_df, top_k=k, verbose=True, cache_dir=cache_dir_str
    )
    print(f"\n  Ablation total wall time: {(time.perf_counter()-t0):.1f}s")

    print("\nAblation summary:")
    print(ablation_df.to_string(index=False))

    ablation_path = out_dir / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n  Ablation results saved -> {ablation_path}")

    if save_plots:
        plot_ablation(ablation_df, out_dir)

    # Clinical safety note
    print("\n" + "=" * 65)
    print("  CLINICAL SAFETY ANALYSIS")
    print("=" * 65)
    print(textwrap.dedent("""
  Is this system acceptable at point-of-care?  Generally NO, in isolation.

  Missing a single critical drug-interaction guideline is clinically
  unacceptable. Recall@10 of 0.85 means 15% of critical documents missed.

  Mitigating recommendations (report only — do NOT implement):
  1. HUMAN-IN-THE-LOOP      : Pharmacist review before acting on results.
  2. SAFETY FILTER LAYER    : Rule-based system for high-risk drug pairs.
  3. TUNE FOR RECALL        : Maximise Recall@K at cost of Precision.
  4. ALERT ON UNCERTAINTY   : Warn when max_score < threshold; escalate.
  5. DOMAIN FINE-TUNING     : Fine-tune SBERT on BioASQ / MedQA datasets.
  6. ADVERSARIAL RED-TEAMING: Systematic testing for dangerous missed retrievals.
    """).strip())


# ---------------------------------------------------------------------------
# Part D — FAISS index benchmarks + scale projection (Method 7)
# ---------------------------------------------------------------------------

def run_part_d(
    data_dir:   Path,
    out_dir:    Path,
    cache_dir:  Path,
    device:     str,
    use_cache:  bool,
    save_plots: bool,
) -> None:
    print("\n" + "=" * 65)
    print("  PART D — FAISS Index Benchmarks + Scale Projection  (Method 7)")
    print("=" * 65)

    corpus_df = pd.read_csv(data_dir / "clinical_corpus.csv")

    # ---- Scale Projection (Task 10) ----
    print("\n[Scale Projection @ 10M documents]")
    analyser    = ScaleAnalyser()
    projections = analyser.project()
    analyser.print_table(projections)

    proj_path = out_dir / "scale_projections.json"
    with open(proj_path, "w") as f:
        json.dump([p.model_dump() for p in projections], f, indent=2)
    print(f"  Scale projections saved -> {proj_path}")

    if save_plots:
        plot_scale_projection(projections)

    # ---- FAISS Index Build + Benchmark ----
    print("\n[FAISS Index Benchmarks — Tasks 7, 8, 9]")
    encoder = EmbeddingEncoder(device=device)

    emb_cache = cache_dir / "corpus_embeddings.npy" if use_cache else None
    dense = DenseRetriever(encoder)
    dense.fit(corpus_df, cache_path=emb_cache)
    embeddings = dense._doc_matrix   # reuse already-computed matrix

    benchmarks = []
    index_configs = [
        (IndexType.IVF_FLAT, "faiss_ivfflat.index"),
        (IndexType.HNSW,     "faiss_hnsw.index"),
        (IndexType.IVF_PQ,   "faiss_ivfpq.index"),
    ]

    for idx_type, cache_name in index_configs:
        print(f"\n  Building {idx_type.value} ...")
        faiss_cache = cache_dir / cache_name if use_cache else None

        retriever = FAISSRetriever(index_type=idx_type, n_clusters=min(256, len(corpus_df) // 4))
        retriever.fit(
            embeddings,
            corpus_df["doc_id"].tolist(),
            corpus_df["title"].tolist(),
            corpus_df["topic"].tolist(),
            cache_path=faiss_cache,
        )

        # Benchmark with 30 random query vectors
        n_test = min(30, len(embeddings))
        test_idx = np.random.choice(len(embeddings), n_test, replace=False)
        test_vecs = embeddings[test_idx]

        bm = retriever.benchmark(test_vecs)
        benchmarks.append(bm)
        print(f"  {bm.index_type.value:<12} "
              f"build={bm.build_time_s:.2f}s  "
              f"query={bm.query_time_ms:.2f}ms  "
              f"mem={bm.memory_mb:.1f}MB")

        # IVF-PQ recall vs exact search
        if idx_type == IndexType.IVF_PQ:
            print("\n  IVF-PQ recall analysis:")
            exact_results = []
            approx_results = []
            for vec in test_vecs[:10]:
                # Exact: numpy dot product
                exact_scores = embeddings @ vec
                exact_top = set(np.argsort(exact_scores)[::-1][:10].tolist())
                exact_results.append(exact_top)

                # Approximate: FAISS
                q = vec.reshape(1, -1)
                _, idxs = retriever._index.search(q, 10)
                approx_top = set(int(i) for i in idxs[0] if i >= 0)
                approx_results.append(approx_top)

            recalls = [len(a & e) / len(e) for a, e in zip(approx_results, exact_results)]
            print(f"  IVF-PQ Recall@10 (vs exact): {np.mean(recalls):.3f} "
                  f"(target: ~0.85-0.90)")

    if benchmarks and save_plots:
        plot_index_benchmarks(benchmarks)

    # ---- Method 7: IVF-PQ search demo ----
    print("\n[Method 7] IVF-PQ search demo")
    ivfpq_cache = cache_dir / "faiss_ivfpq.index" if use_cache else None
    ivfpq_retriever = FAISSRetriever(
        index_type=IndexType.IVF_PQ,
        n_clusters=min(256, len(corpus_df) // 4)
    )
    ivfpq_retriever.fit(
        embeddings,
        corpus_df["doc_id"].tolist(),
        corpus_df["title"].tolist(),
        corpus_df["topic"].tolist(),
        cache_path=ivfpq_cache,
    )

    demo_q = "heart attack treatment aspirin antiplatelet"
    q_vec  = dense.encoder.encode_single(demo_q)
    resp   = ivfpq_retriever.search_vector("demo", demo_q, q_vec, top_k=5)
    print(f"\n  IVF-PQ top-5 for: '{demo_q}'")
    for r in resp.results:
        print(f"    Rank {r.rank}: [{r.topic}] score={r.score:.3f}  {r.title[:55]}")

    # ---- Summary comparison ----
    print("\n[FAISS Index Summary]")
    print(f"  {'Index':<12} {'Build(s)':>10} {'Query(ms)':>12} {'Mem(MB)':>10}")
    print("  " + "-" * 48)
    for bm in benchmarks:
        print(f"  {bm.index_type.value:<12} {bm.build_time_s:>10.2f} "
              f"{bm.query_time_ms:>12.3f} {bm.memory_mb:>10.1f}")


# ---------------------------------------------------------------------------
# Combined final comparison across all methods
# ---------------------------------------------------------------------------

def print_final_summary(all_metrics: list[dict], k: int, out_dir: Path, save_plots: bool) -> None:
    all_metrics_clean = [m for m in all_metrics if m and "mrr" in m]
    if len(all_metrics_clean) < 2:
        return

    print("\n" + "=" * 65)
    print(f"  FINAL COMPARISON — All Methods (K={k})")
    print("=" * 65)
    print_comparison_table(all_metrics_clean, k=k)

    if save_plots:
        plot_metrics_comparison(
            all_metrics_clean, out_dir,
            title=f"All 7 Methods — Metric Comparison (K={k})",
            filename="final_all_methods.png"
        )
        plot_latency_comparison(all_metrics_clean, out_dir, filename="final_latency.png")

    # Save final summary
    summary_path = out_dir / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            [{kk: v for kk, v in m.items() if kk != "per_query"} for m in all_metrics_clean],
            f, indent=2
        )
    print(f"\n  Final summary saved -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir   = Path(args.out)
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_plots = not args.no_plots
    use_cache = not args.no_cache

    # Check dataset exists
    if not (data_dir / "clinical_corpus.csv").exists():
        print("Dataset not found. Run:\n  python data/generate_dataset.py")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Project 4: Semantic Search & Information Retrieval at Scale")
    print(f"  Part: {args.part} | K={args.k} | Device: {args.device}")
    print(f"  Cache: {'ENABLED' if use_cache else 'DISABLED'}")
    print(f"{'='*65}")

    all_metrics = []
    t_total = time.perf_counter()

    if args.part in ("A", "all"):
        metrics_a = run_part_a(
            data_dir, out_dir, cache_dir, args.k,
            use_cache=use_cache, save_plots=save_plots
        )
        all_metrics.extend(metrics_a)

    if args.part in ("B", "all"):
        metrics_b = run_part_b(
            data_dir, out_dir, cache_dir, args.k,
            device=args.device, model=args.model,
            use_cache=use_cache, save_plots=save_plots
        )
        if metrics_b:
            all_metrics.append(metrics_b)

    if args.part in ("C", "all"):
        run_part_c(
            data_dir, out_dir, cache_dir, args.k,
            device=args.device, save_plots=save_plots,
            use_cache=use_cache
        )

    if args.part in ("D", "all"):
        run_part_d(
            data_dir, out_dir, cache_dir,
            device=args.device,
            use_cache=use_cache, save_plots=save_plots
        )

    if args.part == "all" and len(all_metrics) >= 2:
        print_final_summary(all_metrics, args.k, out_dir, save_plots)

    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'='*65}")
    print(f"  All done in {total_elapsed:.1f}s")
    print(f"  Outputs -> {out_dir.resolve()}")
    print(f"  Cache   -> {cache_dir.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
