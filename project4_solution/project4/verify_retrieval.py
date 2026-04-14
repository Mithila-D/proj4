"""
Verify retrieval correctness — show actual results vs ground truth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "starter"))

import pandas as pd
import numpy as np

from embeddings import DenseRetriever, EmbeddingEncoder
from pipeline import MultiStagePipeline
from query_expansion import QueryExpander
from retrieval import BM25Retriever

CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR  = Path(__file__).parent / "data"

corpus_df = pd.read_csv(DATA_DIR / "clinical_corpus.csv")
query_df  = pd.read_csv(DATA_DIR / "query_testset.csv")

# Load cached indices
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

# Pick representative queries
test_queries = [
    "heart attack treatment aspirin antiplatelet",   # Q-000: classic lay->clinical
    "water pill swollen ankles fluid retention",     # Q-018: lay term expansion
    "brain bleed anticoagulation stop restart",      # Q-031: cross-topic
    "cholesterol heart disease prevention",           # Q-004: FAILS for BM25
    "blood thinners surgery stop before operation",   # Q-015: FAILS for BM25
    "sugar tablet metformin side effects stomach",    # Q-024: diabetes lay term
    "kidney failure dialysis when to start",           # Q-045: renal lay term
]

for query in test_queries:
    match = query_df[query_df["query_text"] == query]
    relevant_topics = set()
    if len(match) > 0:
        relevant_topics = set(match.iloc[0]["relevant_topics"].split("; "))

    print(f"\n{'='*80}")
    print(f"QUERY: '{query}'")
    if relevant_topics:
        print(f"GROUND TRUTH: {relevant_topics}")
    print(f"{'='*80}")

    # BM25
    resp = bm25.search("x", query, top_k=K)
    bm25_hits = {r.topic for r in resp.results}
    bm25_recall = len(bm25_hits & relevant_topics) / len(relevant_topics) if relevant_topics else 0
    print(f"\n[BM25]  Recall@{K}={bm25_recall:.0%}")
    for r in resp.results[:5]:
        tag = "✅" if r.topic in relevant_topics else "  "
        print(f"  {tag} [{r.rank}] {r.topic:<35} {r.title[:60]}")
    if relevant_topics and bm25_hits & relevant_topics == set():
        print(f"  ⚠️  ZERO relevant topics retrieved!")

    # BM25 + Expansion
    expansion = expander.expand(query)
    resp_exp = bm25_exp.search("x", expansion.expansion_query, top_k=K)
    exp_hits = {r.topic for r in resp_exp.results}
    exp_recall = len(exp_hits & relevant_topics) / len(relevant_topics) if relevant_topics else 0
    print(f"\n[BM25+Exp]  Recall@{K}={exp_recall:.0%}")
    if expansion.ontology_hits:
        print(f"  Expansion terms: {expansion.ontology_hits}")
    for r in resp_exp.results[:5]:
        tag = "✅" if r.topic in relevant_topics else "  "
        print(f"  {tag} [{r.rank}] {r.topic:<35} {r.title[:60]}")
    if relevant_topics and exp_hits & relevant_topics == set():
        print(f"  ⚠️  ZERO relevant topics retrieved!")

    # SBERT Dense
    resp_sbert = dense.search("x", query, top_k=K)
    sbert_hits = {r.topic for r in resp_sbert.results}
    sbert_recall = len(sbert_hits & relevant_topics) / len(relevant_topics) if relevant_topics else 0
    print(f"\n[SBERT]  Recall@{K}={sbert_recall:.0%}")
    for r in resp_sbert.results[:5]:
        tag = "✅" if r.topic in relevant_topics else "  "
        print(f"  {tag} [{r.rank}] {r.topic:<35} {r.title[:60]}")
    if relevant_topics and sbert_hits & relevant_topics == set():
        print(f"  ⚠️  ZERO relevant topics retrieved!")

    # MultiStage
    resp_multi = pipeline.retrieve("x", query)
    multi_hits = {r.topic for r in resp_multi.results}
    multi_recall = len(multi_hits & relevant_topics) / len(relevant_topics) if relevant_topics else 0
    print(f"\n[MultiStage]  Recall@{K}={multi_recall:.0%}")
    for r in resp_multi.results[:5]:
        tag = "✅" if r.topic in relevant_topics else "  "
        print(f"  {tag} [{r.rank}] {r.topic:<35} {r.title[:60]}")
    if relevant_topics and multi_hits & relevant_topics == set():
        print(f"  ⚠️  ZERO relevant topics retrieved!")

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY — Recall@10 across methods")
print(f"{'='*80}")
rows = []
for query in test_queries:
    match = query_df[query_df["query_text"] == query]
    if len(match) == 0:
        continue
    relevant = set(match.iloc[0]["relevant_topics"].split("; "))

    # BM25
    r = bm25.search("x", query, top_k=K)
    bm25_rec = len({x.topic for x in r.results} & relevant) / len(relevant)

    # BM25+Exp
    exp = expander.expand(query)
    r2 = bm25_exp.search("x", exp.expansion_query, top_k=K)
    exp_rec = len({x.topic for x in r2.results} & relevant) / len(relevant)

    # SBERT
    r3 = dense.search("x", query, top_k=K)
    sbert_rec = len({x.topic for x in r3.results} & relevant) / len(relevant)

    # MultiStage
    r4 = pipeline.retrieve("x", query)
    multi_rec = len({x.topic for x in r4.results} & relevant) / len(relevant)

    rows.append({
        "query": query[:50],
        "relevant": "; ".join(relevant),
        "BM25": bm25_rec,
        "BM25+Exp": exp_rec,
        "SBERT": sbert_rec,
        "MultiStage": multi_rec,
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
print(f"\nMean recall:")
print(f"  BM25:       {summary_df['BM25'].mean():.3f}")
print(f"  BM25+Exp:   {summary_df['BM25+Exp'].mean():.3f}")
print(f"  SBERT:      {summary_df['SBERT'].mean():.3f}")
print(f"  MultiStage: {summary_df['MultiStage'].mean():.3f}")
