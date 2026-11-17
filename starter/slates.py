"""Existing independent per-user recommendation, no shared stock allocation."""
import numpy as np

def recommend(values,slots):
    v=np.asarray(values);return [sorted(range(v.shape[1]),key=lambda j:(-v[u,j],j))[:slots] for u in range(v.shape[0])]
