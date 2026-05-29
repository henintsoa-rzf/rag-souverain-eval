"""scripts/00_inspect_token_lengths.py — distribution de longueur en tokens."""
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("AgentPublic/service-public", cache_dir="data/raw")
data = ds[next(iter(ds.keys()))]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

# Échantillon de 2000 chunks (suffisant pour la distribution)
sample = data["chunk_text"]
lengths = [len(tok.encode(t, add_special_tokens=True)) for t in sample if t]
lengths.sort()
n = len(lengths)

print(f"Échantillon : {n} chunks")
print(f"  min    : {lengths[0]}")
print(f"  P25    : {lengths[n//4]}")
print(f"  median : {lengths[n//2]}")
print(f"  P75    : {lengths[3*n//4]}")
print(f"  P90    : {lengths[int(0.90*n)]}")
print(f"  P95    : {lengths[int(0.95*n)]}")
print(f"  P99    : {lengths[int(0.99*n)]}")
print(f"  max    : {lengths[-1]}")

for threshold in [256, 384, 512, 640, 768, 1024]:
    truncated = sum(1 for l in lengths if l > threshold)
    print(f"  max_seq_length={threshold}: tronque {truncated}/{n} ({truncated*100/n:.1f}%) des chunks")