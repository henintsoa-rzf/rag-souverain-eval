"""
J1 — Bench embedding CPU pour Qwen3-Embedding-0.6B.

Objectif :
  1. Mesurer le throughput (textes/s, tokens/s) sur CPU.
  2. Trouver le batch_size optimal (8/16/32).
  3. Extrapoler la durée d'indexation du corpus complet (~25k chunks).
  4. Décider embedding.device dans default.yaml.

Critère : >50 textes/s sur CPU => device=cpu suffit, ROCm non requis.

Lancer :
  uv run python scripts/00_bench_embedding.py
"""

import time
from statistics import median

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import os

DATASET_NAME = "AgentPublic/service-public"
CACHE_DIR = "data/raw"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
N_TEXTS = 100
BATCH_SIZES = [8, 16, 32]
N_RUNS = 3                       # 1 warmup (jeté) + 2 mesurés -> on prend la médiane des mesurés
DEVICE = "cpu"
TARGET_CORPUS_CHUNKS = 25_000    # estimation post-filtrage


def get_sample_texts(n: int) -> list[str]:
    """Tire n chunks réels représentatifs depuis le dataset."""
    ds = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    data = ds[next(iter(ds.keys()))]
    # On prend chunk_text (le champ réellement embeddé par Etalab) pour réalisme
    texts = data.select(range(n))["chunk_text"]
    return [t for t in texts if t]

def bench_batch_size(model, texts: list[str], batch_size: int) -> dict:
    """Bench un batch_size donné. Retourne throughput médian."""
    durations = []
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)
        tag = "(warmup, jeté)" if run == 0 else ""
        print(f"    run {run+1}/{N_RUNS}: {elapsed:.2f}s {tag}")

    measured = durations[1:]            # jeter le warmup
    med = median(measured)
    texts_per_s = len(texts) / med
    return {
        "batch_size": batch_size,
        "median_s": med,
        "texts_per_s": texts_per_s,
    }


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "12"     # cores physiques du HX370
    os.environ["MKL_NUM_THREADS"] = "12"
    torch.set_num_threads(12)
    print(f"  torch threads: {torch.get_num_threads()}")

    print(f"=== Bench embedding: {MODEL_NAME} on {DEVICE} ===")
    print(f"Échantillon : {N_TEXTS} chunks réels | runs/config : {N_RUNS} (1 warmup jeté)")

    print("\nChargement échantillon...")
    texts = get_sample_texts(N_TEXTS)
    avg_chars = sum(len(t) for t in texts) / len(texts)
    print(f"  {len(texts)} textes, longueur moyenne {avg_chars:.0f} chars")

    print(f"\nChargement modèle {MODEL_NAME} (device={DEVICE})...")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    load_time = time.perf_counter() - t0
    print(f"  Modèle chargé en {load_time:.1f}s | dim={model.get_embedding_dimension()}")

    results = []
    for bs in BATCH_SIZES:
        print(f"\n--- batch_size = {bs} ---")
        results.append(bench_batch_size(model, texts, bs))

    # ===== Synthèse =====
    print("\n" + "=" * 55)
    print("SYNTHÈSE")
    print("=" * 55)
    print(f"{'batch_size':>11} | {'médiane (s)':>11} | {'textes/s':>10} | {'corpus 25k':>12}")
    print("-" * 55)
    best = max(results, key=lambda r: r["texts_per_s"])
    for r in results:
        eta_min = TARGET_CORPUS_CHUNKS / r["texts_per_s"] / 60
        flag = " <-- best" if r is best else ""
        print(
            f"{r['batch_size']:>11} | {r['median_s']:>11.2f} | "
            f"{r['texts_per_s']:>10.1f} | {eta_min:>9.1f} min{flag}"
        )

    print("\n--- DÉCISION ---")
    if best["texts_per_s"] >= 50:
        print(f"  ✅ {best['texts_per_s']:.0f} textes/s >= 50 => device: cpu SUFFIT")
        print(f"     Indexation ~25k chunks : ~{TARGET_CORPUS_CHUNKS/best['texts_per_s']/60:.0f} min")
        print(f"     ROCm/iGPU : reporté (non nécessaire). Figer device=cpu, batch_size={best['batch_size']}.")
    else:
        print(f"  ⚠️  {best['texts_per_s']:.0f} textes/s < 50 => CPU lent")
        print(f"     Indexation ~25k chunks : ~{TARGET_CORPUS_CHUNKS/best['texts_per_s']/60:.0f} min")
        print("     Acceptable si one-shot. Sinon envisager iGPU ROCm en S2.")

    print("\n=== Bench terminé ===")


if __name__ == "__main__":
    main()

