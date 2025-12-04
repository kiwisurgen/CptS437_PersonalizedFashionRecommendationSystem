import json, numpy as np, os
emb="data/embeddings/clip_embeddings.npy"
meta="data/embeddings/embedding_metadata.json"
print("emb exists:", os.path.exists(emb))
if os.path.exists(emb):
    a=np.load(emb)
    print("emb shape:", a.shape, "dtype:", a.dtype)
    norms = np.linalg.norm(a, axis=1)
    print("norms: min", norms.min(), "max", norms.max(), "mean", norms.mean())
if os.path.exists(meta):
    m=json.load(open(meta))
    print("meta count:", m.get('count'), "ids:", len(m.get('ids',[])))
    print("sample ids:", m.get('ids',[])[:5])