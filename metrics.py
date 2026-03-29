"""
Metrics:
  1. topic_hit_rate     — topic IoU overlap (predicted vs actual rubrics/clusters)
  2. entity_match_rate  — entity recall (persons, orgs)
  3. semantic_similarity— cosine sim via sentence-transformers
  4. style_match        — TF-IDF cosine vs outlet corpus average
  5. diversity_score    — 1 − mean pairwise similarity among predictions
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTLET_SLUGS
from backtester import load_backtest
from analyzer import (
    _get_stopwords, _texts_for_outlet, extract_entities,
)

# ================================================================
#  SENTENCE TRANSFORMER (lazy-loaded)
# ================================================================
_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print("  [metrics] sentence-transformers model loaded")
        except Exception as exc:
            print(f"  [metrics] sentence-transformers not available: {exc}")
    return _ST_MODEL


# ================================================================
#  1. TOPIC HIT RATE
# ================================================================

def topic_hit_rate(pred_topics: List[str], actual_topics: List[str]) -> float:
    """
    Jaccard IoU on token-level topic overlap.
    Both lists are token-sets of topic labels (words from cluster names).
    """
    def _tokenize(topics: List[str]) -> set:
        tokens = set()
        for t in topics:
            tokens.update(re.findall(r"[а-яёА-ЯЁa-zA-Z]{3,}", t.lower()))
        return tokens

    pred_set   = _tokenize(pred_topics)
    actual_set = _tokenize(actual_topics)
    if not actual_set:
        return 0.0
    intersection = pred_set & actual_set
    union        = pred_set | actual_set
    return len(intersection) / len(union) if union else 0.0


# ================================================================
#  2. ENTITY MATCH RATE
# ================================================================

def entity_match_rate(pred_texts: List[str], actual_texts: List[str]) -> float:
    """
    Precision/Recall F1 of named entity overlap (persons + orgs).
    """
    if not actual_texts or not pred_texts:
        return 0.0

    pred_ents   = set()
    actual_ents = set()

    for ent_dict in [extract_entities(pred_texts), extract_entities(actual_texts)]:
        pass  # iterate below

    p_ents = extract_entities(pred_texts)
    a_ents = extract_entities(actual_texts)

    for key in ("persons", "orgs"):
        pred_ents.update([e.lower() for e in p_ents.get(key, [])])
        actual_ents.update([e.lower() for e in a_ents.get(key, [])])

    if not actual_ents:
        return 0.0

    matched = pred_ents & actual_ents
    precision = len(matched) / len(pred_ents)   if pred_ents   else 0.0
    recall    = len(matched) / len(actual_ents) if actual_ents else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ================================================================
#  3. SEMANTIC SIMILARITY
# ================================================================

def semantic_similarity(pred_texts: List[str], actual_texts: List[str]) -> float:
    """
    Average pairwise cosine similarity between predicted and actual texts
    using multilingual sentence embeddings.
    """
    model = _get_st_model()
    if model is None or not pred_texts or not actual_texts:
        return 0.0

    try:
        pred_emb   = model.encode(pred_texts,   convert_to_numpy=True, show_progress_bar=False)
        actual_emb = model.encode(actual_texts, convert_to_numpy=True, show_progress_bar=False)
        sims = cosine_similarity(pred_emb, actual_emb)
        # Mean of max similarity per predicted text
        return float(np.mean(sims.max(axis=1)))
    except Exception as exc:
        print(f"  [metrics] semantic_similarity error: {exc}")
        return 0.0


# ================================================================
#  4. STYLE MATCH
# ================================================================

def style_match(pred_texts: List[str], corpus_texts: List[str]) -> float:
    """
    TF-IDF cosine similarity between predicted texts and outlet corpus average.
    """
    if not pred_texts or not corpus_texts:
        return 0.0

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=5000,
        stop_words=list(stopwords),
    )
    try:
        all_texts   = corpus_texts + pred_texts
        tfidf       = vec.fit_transform(all_texts)
        corpus_vecs = tfidf[:len(corpus_texts)]
        pred_vecs   = tfidf[len(corpus_texts):]
        corpus_mean = np.asarray(corpus_vecs.mean(axis=0))
        sims = cosine_similarity(pred_vecs, corpus_mean)
        return float(sims.mean())
    except Exception as exc:
        print(f"  [metrics] style_match error: {exc}")
        return 0.0


# ================================================================
#  5. DIVERSITY SCORE
# ================================================================

def diversity_score(pred_texts: List[str]) -> float:
    """
    1 − mean pairwise cosine similarity among predictions.
    Higher = more diverse.
    """
    if len(pred_texts) < 2:
        return 1.0

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=3000,
        stop_words=list(stopwords),
    )
    try:
        tfidf = vec.fit_transform(pred_texts)
        sims  = cosine_similarity(tfidf)
        # Exclude diagonal
        n = sims.shape[0]
        off_diag = sims[np.triu_indices(n, k=1)]
        return float(1 - off_diag.mean()) if len(off_diag) > 0 else 1.0
    except Exception as exc:
        print(f"  [metrics] diversity_score error: {exc}")
        return 0.0


# ================================================================
#  FULL REPORT
# ================================================================

def full_report(backtest_results: List[Dict], outlet: str,
                corpus_texts: Optional[List[str]] = None,
                method: str = "frequency") -> Dict:
    """
    Compute all 5 metrics across backtest days for a given method.
    """
    from etl import load_clean

    if corpus_texts is None:
        df = load_clean(outlet)
        corpus_texts = df["title"].dropna().tolist()

    topic_hits: List[float]   = []
    entity_ms:  List[float]   = []
    sem_sims:   List[float]   = []
    style_ms:   List[float]   = []
    div_scores: List[float]   = []

    for day_res in backtest_results:
        preds   = day_res.get("predictions", {}).get(method, [])
        actual  = day_res.get("actual", {})

        pred_topics  = [p.get("topic_label", "") for p in preds]
        pred_titles  = [p["title"] for p in preds if p.get("title")]
        actual_topics= actual.get("rubrics", [])
        actual_titles= actual.get("titles", [])

        if not actual_titles:
            continue

        topic_hits.append(topic_hit_rate(pred_topics, actual_topics))
        entity_ms.append(entity_match_rate(pred_titles, actual_titles))
        sem_sims.append(semantic_similarity(pred_titles, actual_titles))
        style_ms.append(style_match(pred_titles, corpus_texts))
        div_scores.append(diversity_score(pred_titles))

    def _mean(lst): return round(float(np.mean(lst)), 4) if lst else 0.0

    report = {
        "outlet":              outlet,
        "method":              method,
        "days_evaluated":      len(topic_hits),
        "topic_hit_rate":      _mean(topic_hits),
        "entity_match_f1":     _mean(entity_ms),
        "semantic_similarity": _mean(sem_sims),
        "style_match":         _mean(style_ms),
        "diversity_score":     _mean(div_scores),
    }

    print(f"\n[metrics] {outlet} | method={method}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")

    return report


def evaluate_all(slugs: List[str] = None,
                 methods: List[str] = None) -> Dict[str, Dict]:
    if slugs is None:
        slugs = OUTLET_SLUGS
    if methods is None:
        methods = ["inertia", "frequency", "calendar"]

    all_reports: Dict[str, Dict] = {}
    for slug in slugs:
        bt = load_backtest(slug)
        if not bt:
            print(f"  [metrics] No backtest data for {slug}")
            continue
        # Use best baseline method (frequency usually wins)
        best_method = methods[1] if len(methods) > 1 else methods[0]
        report = full_report(bt, slug, method=best_method)
        all_reports[slug] = report
    return all_reports
