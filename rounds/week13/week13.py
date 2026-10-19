"""Capstone BBO - Module 24 (Week 13, FINAL round) query selection.

Final-round policy: pure exploitation. With no later round, an exploratory query has zero
information value, so every function is sent to the point with the strongest evidence:
  - noise-free functions (F1, F4, F5, F7, F8): the highest observed output. This automatically
    keeps a round-12 single-coordinate probe only if it beat the incumbent.
  - noisy functions (F2, F3, F6): identical inputs are grouped and averaged, and points are ranked
    by a lower confidence bound, mean - sd/sqrt(n), with sd pooled from the replicated points.
    A single lucky reading is therefore not trusted over a replicated one.
The rule reads data/processed/*.csv, so it re-runs unchanged once the round-12 results are added.
"""
import numpy as np, pandas as pd
NOISY = {2, 3, 6}

def best_point(i):
    d = pd.read_csv(f'../../data/processed/function_{i}.csv')
    xc = list(d.filter(regex=r'^x\d+$').columns)
    d['key'] = d[xc].round(6).astype(str).agg('-'.join, axis=1)
    g = d.groupby('key').agg(y_mean=('y', 'mean'), n=('y', 'size'), y_max=('y', 'max'))
    if i in NOISY:
        sd = d.groupby('key').y.std().dropna()           # replicated points only
        sd = float(np.sqrt((sd ** 2).mean())) if len(sd) else 0.0
        g['score'] = g.y_mean - sd / np.sqrt(g.n)
    else:
        g['score'] = g.y_max
    g = g.sort_values('score', ascending=False)
    k = g.index[0]
    x = d.loc[d.key == k, xc].iloc[0].to_numpy()
    return np.clip(x, 1e-6, 1 - 1e-6), g.iloc[0], None

rows = []
for i in range(1, 9):
    x, stats, _ = best_point(i)
    q = '-'.join(f'{v:.6f}' for v in x)
    assert all(len(p) == 8 and p.startswith('0.') for p in q.split('-')) and len(q.split('-')) == len(x)
    rows.append((i, q))
    print(f'F{i}  {q}   mean {stats.y_mean:.6g}  n={int(stats.n)}  max {stats.y_max:.6g}')
pd.DataFrame(rows, columns=['function', 'query_string']).to_csv('Week13_queries.csv', index=False)
