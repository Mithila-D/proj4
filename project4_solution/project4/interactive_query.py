"""
Interactive Query Tool — Test your own queries against all retrieval methods.
=============================================================================
Run:
    python interactive_query.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "starter"))

import json
import numpy as np
import pandas as pd

from embeddings import DenseRetriever, EmbeddingEncoder
from pipeline import MultiStagePipeline
from query_expansion import QueryExpander
from retrieval import BM25Retriever, TFIDFRetriever

CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR  = Path(__file__).parent / "data"

def main():
    corpus_df = pd.read_csv(DATA_DIR / "clinical_corpus.csv")
    query_df  = pd.read_csv(DATA_DIR / "query_testset.csv")

    print("=" * 60)
    print("  Interactive Query Tool")
    print("  Type a query (or 'quit' to exit)")
    print("=" * 60)

    # ---- Load cached indices ----
    print("\nLoading cached indices...")
    tfidf = TFIDFRetriever()
    tfidf.fit(corpus_df, cache_path=CACHE_DIR / "tfidf.pkl")

    bm25 = BM25Retriever()
    bm25.fit(corpus_df, cache_path=CACHE_DIR / "bm25.pkl")

    expander = QueryExpander()

    encoder = EmbeddingEncoder()
    dense = DenseRetriever(encoder)
    dense.fit(corpus_df, cache_path=CACHE_DIR / "corpus_embeddings.npy")

    bm25_exp = BM25Retriever()
    bm25_exp.fit(corpus_df)

    pipeline = MultiStagePipeline(expander, bm25_exp, encoder, candidate_k=100)
    pipeline.index_corpus(corpus_df, doc_embeddings=dense._doc_matrix)

    K = 10

    while True:
        query = input("\n🔍 Query > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not query:
            continue

        print(f"\n{'='*60}")
        print(f"  Query: '{query}'")
        print(f"{'='*60}")

        # 1. BM25
        resp_bm25 = bm25.search("q1", query, top_k=K)
        print(f"\n[BM25] Top-{K}:")
        for r in resp_bm25.results:
            print(f"  [{r.rank}] {r.title[:70]}  ({r.topic})  score={r.score:.3f}")

        # 2. BM25 + Expansion
        expansion = expander.expand(query)
        print(f"  Expansion hits: {list(expansion.ontology_hits.keys())}")
        print(f"  Added terms: {[t for t in expansion.added_terms]}")
        resp_exp = bm25_exp.search("q1", expansion.expansion_query, top_k=K)
        print(f"\n[BM25+Exp] Top-{K}:")
        for r in resp_exp.results:
            print(f"  [{r.rank}] {r.title[:70]}  ({r.topic})  score={r.score:.3f}")

        # 3. SBERT Dense
        resp_sbert = dense.search("q1", query, top_k=K)
        print(f"\n[SBERT Dense] Top-{K}:")
        for r in resp_sbert.results:
            print(f"  [{r.rank}] {r.title[:70]}  ({r.topic})  score={r.score:.3f}")

        # 4. MultiStage
        resp_multi = pipeline.retrieve("q1", query)
        print(f"\n[MultiStage] Top-{K}:")
        for r in resp_multi.results:
            print(f"  [{r.rank}] {r.title[:70]}  ({r.topic})  score={r.score:.3f}")

        # --- Check against ground truth if query matches testset ---
        match = query_df[query_df["query_text"].str.lower() == query.lower()]
        if len(match) > 0:
            relevant = set(match.iloc[0]["relevant_topics"].split("; "))
            bm25_topics = {r.topic for r in resp_bm25.results}
            exp_topics = {r.topic for r in resp_exp.results}
            sbert_topics = {r.topic for r in resp_sbert.results}
            multi_topics = {r.topic for r in resp_multi.results}

            print(f"\n{'─'*60}")
            print(f"  Ground truth relevant topics: {relevant}")
            print(f"  BM25 recall@{K}:       {len(bm25_topics & relevant)}/{len(relevant)}")
            print(f"  BM25+Exp recall@{K}:   {len(exp_topics & relevant)}/{len(relevant)}")
            print(f"  SBERT recall@{K}:      {len(sbert_topics & relevant)}/{len(relevant)}")
            print(f"  MultiStage recall@{K}: {len(multi_topics & relevant)}/{len(relevant)}")

if __name__ == "__main__":
    main()
