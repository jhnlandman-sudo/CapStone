"""Capstone BBO - Module 22 (Week 11) query selection.  20 portal rounds of data.

Round-10 outcome (written prediction -> actual):
  F1  centre                   2.0                -> 2.0           noise-free, solved.
  F2  replicate #3             0.69 / GP 0.610    -> 0.597542      three readings at one point: 0.650, 0.726, 0.598
                                                                    (mean 0.658, sd 0.064).  Noise is as large as any
                                                                    local signal F2 has ever shown.
  F3  replicate #2             -0.026 if real     -> -0.046383     the -0.0258 record was a lucky draw (mean -0.036).
  F4  x4 = 0.425034            0.66 +/- 0.03      -> 0.708891      NEW BEST.  The x4 peak lies below 0.428, not at it.
  F5  corner                   8662.405           -> 8662.405001   noise-free, solved.
  F6  replicate #2             -0.061 if real     -> -0.087772     the peak is real (mean -0.074); neighbours 0.03-0.05
                                                                    away return -0.26 to -0.29, so it is narrow.
  F7  GP mean-max r0.08        2.1988 +/- 0.0588  -> 2.301303      NEW BEST (+0.133, +1.7 sd above prediction).
  F8  dim4 -> 0                9.9178 +/- 0.009   -> 9.902873      FALSIFIED: dim 4 belongs above 0.0388, not below.

Policy this round:
  F1, F5  resubmit the known optimum.
  F4  three-point x4 bracket (0.425/0.428/0.433) -> vertex 0.416634, predicted 0.743 (curvature -485).
  F6  x2 bracket along the path that produced the peak (0.25 / 0.365 / 0.414) -> vertex 0.341626, others held.
  F7  repeat what worked: GP posterior-mean maximiser, radius 0.08 around the new incumbent.
  F8  dim 4 the other way: GP line scan peaks above, take a moderate +0.10 step to 0.138786.
  F2  x2 barely matters: every reading with x1 in 0.66-0.70 lies in 0.54-0.73 whatever x2 is.
      F2 is a ridge in x1.  Quadratic in x1 over the ridge readings -> 0.691064, x2 held at 0.926564.
  F3  GP posterior-mean maximiser, radius 0.05 (converged to within noise).
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
S = json.load(open('gp_settings.json'))
rng = np.random.default_rng(22)

def load(i):
    d = pd.read_csv(f'../../data/processed/function_{i}.csv')
    return d.filter(regex=r'^x\d+$').to_numpy(), d.y.to_numpy()
def gp_for(i):
    X, y = load(i); z = np.log(y) if i == 5 else y; d, s = X.shape[1], S[str(i)]
    base = (RBF(np.full(d, .3), (s['ls_lo'], 2.)) if s['kernel'] == 'rbf'
            else Matern(np.full(d, .3), (s['ls_lo'], 2.), nu=2.5))
    k = ConstantKernel(1., (1e-3, 1e3)) * base + WhiteKernel(1e-4, (1e-8, 1.))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=25,
                                    random_state=0).fit(X, z), X, y
def inc(i): X, y = load(i); return X[np.argmax(y)].copy()
def probe(i, ch):
    x = inc(i)
    for d, v in ch.items(): x[d] = v
    return np.clip(x, 1e-6, 1 - 1e-6)
def vertex(xs, ys):
    c = np.polyfit(xs, ys, 2); v = -c[1] / (2 * c[0]); return v, np.polyval(c, v)
def mean_max(i, rad, n=200000):
    g, X, y = gp_for(i); x0 = inc(i)
    C = rng.uniform(np.maximum(0, x0 - rad), np.minimum(1, x0 + rad), (n, X.shape[1]))
    mu, sd = g.predict(C, return_std=True); j = np.argmax(mu)
    return np.clip(C[j], 1e-6, 1 - 1e-6), f"GP {mu[j]:.4f} +/- {sd[j]:.4f}"
def gp_pred(i, x):
    g, *_ = gp_for(i); m, s = g.predict(x[None], return_std=True); return f"GP {m[0]:.4f} +/- {s[0]:.4f}"

Q, P = {}, {}
Q[1] = inc(1); P[1] = "2.0"
Q[5] = inc(5); P[5] = "8662.405"
v4, p4 = vertex([0.425034, 0.428034, 0.433034], [0.708891, 0.680110, 0.612760])
Q[4] = probe(4, {3: v4}); P[4] = f"{p4:.4f} (quadratic)"
v6, p6 = vertex([0.25, 0.365418, 0.4141], [-0.439094, -0.074, -0.292534])
Q[6] = probe(6, {1: v6}); P[6] = f"{p6:.3f} +/- 0.03 (quadratic, others held)"
Q[7], P[7] = mean_max(7, 0.08)
Q[8] = probe(8, {3: 0.138786}); P[8] = gp_pred(8, Q[8])
# F2: the data form a ridge in x1 (0.66-0.70 gives 0.54-0.73 at every x2 from 0.12 to 1.0;
# x1 = 0.58 or 0.74 collapses).  Fit a quadratic in x1 to the ridge readings (repeats averaged),
# keep x2 at the best-read value.
x1 = [0.6658, 0.681098, 0.682637, 0.702637, 0.703, 0.739667]
y1 = [0.538996, 0.569387, 0.657841, 0.611205, 0.573418, 0.359876]
c = np.polyfit(x1, y1, 2); v2 = -c[1] / (2 * c[0])
Q[2] = np.array([v2, 0.926564]); P[2] = f"{np.polyval(c, v2):.3f} ridge mean, noise sd ~0.06"
Q[3], P[3] = mean_max(3, 0.05)

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
rows = []
for i in range(1, 9):
    q = "-".join(f"{v:.6f}" for v in Q[i]); parts = q.split("-")
    assert len(parts) == DIMS[i] and all(p.startswith("0.") and len(p) == 8 for p in parts), (i, q)
    rows.append(f"{i},{q}"); print(f"F{i}: {q}    pred {P[i]}")
open("Week11_queries.csv", "w").write("function,query_string\n" + "\n".join(rows) + "\n")
