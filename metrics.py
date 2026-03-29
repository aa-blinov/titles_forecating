"""
Metrics:
  1. topic_hit_rate     — topic IoU overlap (predicted vs actual rubrics/clusters)
  2. entity_match_rate  — entity recall (persons, orgs)
  3. semantic_similarity— cosine sim via sentence-transformers
  4. style_match        — TF-IDF cosine vs outlet corpus average
  5. diversity_score    — 1 − mean pairwise similarity among predictions
"""
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTLET_SLUGS, OPENROUTER_API_KEY, OPENROUTER_URL
from analyzer import (
    _get_stopwords, _texts_for_outlet, extract_entities,
)

# Load helper to handle circular imports if any (though lazy is safer)
def load_backtest_internal(outlet: str) -> List[Dict]:
    from backtester import load_backtest
    return load_backtest(outlet)

# ================================================================
#  OPENROUTER EMBEDDER (API-based)
# ================================================================
_EMB_CACHE = {}

def _get_openrouter_embeddings(texts: Tuple[str]) -> np.ndarray:
    if not OPENROUTER_API_KEY:
        print("  [metrics] OPENROUTER_API_KEY missing, semantic similarity will be 0")
        return None
        
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=OPENROUTER_URL,
            api_key=OPENROUTER_API_KEY,
        )
        # Используется модель Qwen3-8b через OpenRouter:
        resp = client.embeddings.create(
            model="qwen/qwen3-embedding-8b",
            input=list(texts)
        )
        embeddings = [item.embedding for item in resp.data]
        return np.array(embeddings)
    except Exception as exc:
        print(f"  [metrics] OpenRouter Embeddings Error: {exc}")
        return None


def _encode_texts(texts: Tuple[str]):
    if not texts:
        return None
    key = hash(texts)
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = _get_openrouter_embeddings(texts)
    return _EMB_CACHE[key]


# ================================================================
#  1. TOPIC HIT RATE
# ================================================================

def topic_hit_rate(pred_topics: List[str], actual_topics: List[str]) -> float:
    """
    Jaccard IoU on token-level topic overlap.
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
    Semantic F1 score using pairwise cosine similarity between embeddings.
    Measures both Precision (are predictions real-like?) and Recall (are actuals covered?).
    """
    if not pred_texts or not actual_texts:
        return 0.0

    try:
        pred_emb   = _encode_texts(tuple(pred_texts))
        actual_emb = _encode_texts(tuple(actual_texts))
        
        if pred_emb is None or actual_emb is None:
            return 0.0
            
        sims = cosine_similarity(pred_emb, actual_emb)
        
        precision = float(np.mean(sims.max(axis=1)))  # Each pred to its best actual
        recall    = float(np.mean(sims.max(axis=0)))  # Each actual to its best pred
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    except Exception as exc:
        print(f"  [metrics] semantic_similarity error: {exc}")
        return 0.0


# ================================================================
#  4. STYLE MATCH
# ================================================================

_STYLE_CACHE = {}

def style_match(pred_texts: List[str], corpus_texts: List[str]) -> float:
    """
    TF-IDF cosine similarity between predicted texts and outlet corpus average.
    """
    if not pred_texts or not corpus_texts:
        return 0.0

    global _STYLE_CACHE
    cache_key = (len(corpus_texts), corpus_texts[0] if corpus_texts else "")
    
    try:
        if cache_key not in _STYLE_CACHE:
            stopwords = _get_stopwords()
            vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                max_features=5000,
                stop_words=list(stopwords),
            )
            
            texts = corpus_texts
            # Берем первые 2000 - это быстрее и стабильнее для кэширования
            if len(texts) > 2000:
                texts = texts[:2000]
                
            # Fit ONLY on the real corpus to prevent data leakage
            corpus_vecs = vec.fit_transform(texts)
            corpus_mean = np.asarray(corpus_vecs.mean(axis=0))
            _STYLE_CACHE[cache_key] = (vec, corpus_mean)
            
        vec, corpus_mean = _STYLE_CACHE[cache_key]
        pred_vecs   = vec.transform(pred_texts)
        
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
    1 − mean pairwise similarity among predictions.
    """
    if len(pred_texts) < 2:
        return 1.0

    stopwords = _get_stopwords()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=3000,
    )
    try:
        tfidf = vec.fit_transform(pred_texts)
        sims  = cosine_similarity(tfidf)
        n = sims.shape[0]
        off_diag = sims[np.triu_indices(n, k=1)]
        return float(1 - off_diag.mean()) if len(off_diag) > 0 else 1.0
    except Exception as exc:
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

    return report


def evaluate_all(slugs: List[str] = None,
                 methods: List[str] = None) -> List[Dict]:
    """
    Generate reports for all outlets and all available methods.
    """
    if slugs is None:
        slugs = OUTLET_SLUGS
    if methods is None:
        methods = ["inertia", "frequency", "calendar", "llm"]

    all_reports: Dict[str, Dict] = {}
    from etl import load_clean
    from tqdm import tqdm

    print("\n[metrics] ================= STARTING EVALUATION =================")

    for slug in slugs:
        bt = load_backtest_internal(slug)
        if not bt:
            print(f"  [metrics] ⚠️ No backtest data for {slug}")
            continue
            
        df = load_clean(slug)
        corpus_texts = df["title"].dropna().tolist()
        
        # Check available methods in the first successful day
        available_methods = []
        for d in bt:
            if d.get("predictions"):
                available_methods = list(d["predictions"].keys())
                break
        
        methods_to_run = [m for m in available_methods if m in methods]
        if not methods_to_run:
            print(f"  [metrics] ⏭️ Skipping {slug.upper()}: No active methods to test.")
            continue
            
        print(f"\n📰 Outlet: {slug.upper()} | Base texts: {len(corpus_texts)} | Methods: {len(methods_to_run)}")
        
        for method in tqdm(methods_to_run, desc=f"Evaluating", leave=False):
            report = full_report(bt, slug, corpus_texts=corpus_texts, method=method)
            key = f"{slug} ({method})"
            all_reports[key] = report
            
    print("\n[metrics] =================== EVALUATION DONE ===================\n")
    return all_reports
