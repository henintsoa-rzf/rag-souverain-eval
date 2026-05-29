"""
J1 — Bench embedding CPU v2.

Changements vs v1 :
  - Multi-threading PyTorch forcé (cores physiques)
  - max_seq_length=512 explicite (truncation contrôlée)
  - Comparaison Qwen3-Embedding-0.6B vs BGE-M3 (baseline d'ablation)
  - Mesure aussi tokens/s (plus comparable entre modèles)

Lancer :
  uv run python scripts/00_bench_embedding_v2.py 2>&1 | tee data/raw/bench_embedding_v2_report.txt
"""

import os
import time
from statistics import median

# IMPORTANT: forcer le multi-threading AVANT d'importer torch
N_CORES = 12  # cores physiques HX470
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

import torch
torch.set_num_threads(N_CORES)

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATASET_NAME = "AgentPublic/service-public"
CACHE_DIR = "data/raw"
MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "qwen3"),
    ("BAAI/bge-m3",               "bgem3"),
]
N_TEXTS = 100
BATCH_SIZES = [16, 32]
N_RUNS = 3                       # 1 warmup + 2 mesurés
MAX_SEQ_LENGTH = 768
DEVICE = "cpu"
TARGET_CORPUS_CHUNKS = 25_000


def get_sample_texts(n: int) -> list[str]:
    ds = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    data = ds[next(iter(ds.keys()))]
    texts = data.select(range(n))["chunk_text"]
    return [t for t in texts if t]


def bench_model(model_name: str, short_name: str, texts: list[str]) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Modèle : {model_name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=DEVICE, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"  Chargé en {time.perf_counter()-t0:.1f}s | "
          f"dim={model.get_embedding_dimension()} | "
          f"max_seq_len={model.max_seq_length}")

    results = []
    for bs in BATCH_SIZES:
        print(f"\n--- batch_size = {bs} ---")
        durations = []
        for run in range(N_RUNS):
            t0 = time.perf_counter()
            model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            elapsed = time.perf_counter() - t0
            durations.append(elapsed)
            tag = "(warmup)" if run == 0 else ""
            print(f"    run {run+1}/{N_RUNS}: {elapsed:.2f}s {tag}")

        med = median(durations[1:])
        results.append({
            "model": short_name,
            "batch_size": bs,
            "median_s": med,
            "texts_per_s": len(texts) / med,
        })

    # Libère la RAM avant le modèle suivant
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return results


def main() -> None:
    print(f"=== Bench embedding v2 (device={DEVICE}) ===")
    print(f"torch threads : {torch.get_num_threads()}")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"Échantillon   : {N_TEXTS} chunks | runs : {N_RUNS} (1 warmup jeté)")

    texts = get_sample_texts(N_TEXTS)
    avg_chars = sum(len(t) for t in texts) / len(texts)
    print(f"  {len(texts)} textes, longueur moyenne {avg_chars:.0f} chars")

    all_results = []
    for model_name, short in MODELS:
        all_results.extend(bench_model(model_name, short, texts))

    # ===== Synthèse =====
    print("\n" + "="*70)
    print("SYNTHÈSE")
    print("="*70)
    print(f"{'modèle':>8} | {'batch':>5} | {'médiane s':>10} | {'textes/s':>9} | {'corpus 25k':>12}")
    print("-"*70)
    for r in all_results:
        eta_min = TARGET_CORPUS_CHUNKS / r["texts_per_s"] / 60
        print(
            f"{r['model']:>8} | {r['batch_size']:>5} | "
            f"{r['median_s']:>10.2f} | {r['texts_per_s']:>9.1f} | "
            f"{eta_min:>9.1f} min"
        )

    # Décision
    best_qwen = max((r for r in all_results if r["model"] == "qwen3"), key=lambda r: r["texts_per_s"])
    best_bgem3 = max((r for r in all_results if r["model"] == "bgem3"), key=lambda r: r["texts_per_s"])
    print("\n--- DÉCISION ---")
    print(f"  Qwen3 best : {best_qwen['texts_per_s']:.1f} t/s @ bs={best_qwen['batch_size']}")
    print(f"  BGE-M3 best: {best_bgem3['texts_per_s']:.1f} t/s @ bs={best_bgem3['batch_size']}")
    speedup = best_bgem3["texts_per_s"] / best_qwen["texts_per_s"]
    print(f"  Ratio BGE-M3/Qwen3 : x{speedup:.1f}")
    print(f"\n  Qwen3 indexation 25k : ~{TARGET_CORPUS_CHUNKS/best_qwen['texts_per_s']/60:.0f} min")
    print(f"  (rappel: BGE-M3 baseline = 0 min, embeddings déjà fournis par Etalab)")


if __name__ == "__main__":
    main()