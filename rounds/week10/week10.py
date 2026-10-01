"""Capstone BBO - Module 21 (Week 10) query selection.  19 portal rounds of data.

Round-9 outcome (written prediction -> actual):
  F1  centre              2.0              -> 2.0            confirmed, noise-free (3 identical returns)
  F2  exact replicate     0.59             -> 0.725670       FALSIFIED: same input gave 0.650 then 0.726.
                                                              F2 is noisy (GP noise sd now 0.055), and the
                                                              incumbent is genuinely above its neighbours.
  F3  GP mean-max r0.05   -0.0268+/-0.0095 -> -0.043090      FALSIFIED (1.7 sd).  Second GP-mean miss.
  F4  x4 +0.005           0.68+/-0.03      -> 0.612760       FALSIFIED: x4 curvature ~ -2700, sharpest yet.
  F5  replicate corner    8662.405         -> 8662.405001     confirmed exactly: F5 is noise-free.
  F6  GP mean-max r0.08   -0.268 (GP)      -> -0.260729      GP level right; incumbent -0.061 still best.
  F7  x4 = 0.28           2.1674+/-0.0136  -> 2.168082       confirmed. NEW BEST (+0.046).
  F8  half-step, 8 dims   9.9329+/-0.0151  -> 9.911816       FALSIFIED (-1.4 sd); multi-dim GP moves on F8
                                                              have now failed twice.

Policy this round, by function class:
  Solved / noise-free at bound (F1, F5): resubmit.
  Noisy (F2, F3, F6): REPLICATE the best point.  On a noisy function the record is one draw; the
      decision for the final rounds depends on whether it is real.  F2 now has two readings
      (0.650, 0.726); F3's -0.0258 and F6's -0.061 each rest on a single reading.
  Noise-free, still improving (F7): x4 is closed (4-point quadratic, residuals < 1e-3, vertex 0.2885
      worth only +0.0015), so take the GP mean-max in radius 0.08 - its line predictions on F7
      have been accurate to 0.001.
  Noise-free, converged (F4, F8): single-coordinate moves only, so the result is attributable.
      F4: x4 = 0.425034, the third point that makes the x4 vertex computable.
      F8: dim 4 to its lower bound, the best single-coordinate GP gain (+0.005).
"""
import numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
rng = np.random.default_rng(21)
S = json.load(open('gp_settings.json'))

def load(i): return np.load(f'data_w10/f{i}_X.npy'), np.load(f'data_w10/f{i}_y.npy')
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
def gp_pred(i, x):
    g, *_ = gp_for(i); m, s = g.predict(x[None], return_std=True); return f"{m[0]:.4f} +/- {s[0]:.4f}"

Q, PRED = {}, {}
Q[1] = inc(1); PRED[1] = "2.0"
Q[2] = inc(2); PRED[2] = "0.69 (mean of 0.650, 0.726); GP " + gp_pred(2, Q[2])
Q[3] = inc(3); PRED[3] = "-0.026 if the record is real; GP " + gp_pred(3, Q[3])
Q[4] = probe(4, {3: 0.425034}); PRED[4] = "0.66 +/- 0.03 (0.656 if x4 peak is at 0.428)"
Q[5] = inc(5); PRED[5] = "8662.405"
Q[6] = inc(6); PRED[6] = "-0.061 if real; GP " + gp_pred(6, Q[6])
g7, X7, y7 = gp_for(7); x7 = inc(7)
C = rng.uniform(np.maximum(0, x7 - .08), np.minimum(1, x7 + .08), (200000, 6))
Q[7] = np.clip(C[np.argmax(g7.predict(C))], 1e-6, 1 - 1e-6); PRED[7] = gp_pred(7, Q[7])
Q[8] = probe(8, {3: 0.000001}); PRED[8] = gp_pred(8, Q[8])

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
rows = []
for i in range(1, 9):
    q = "-".join(f"{v:.6f}" for v in Q[i]); parts = q.split("-")
    assert len(parts) == DIMS[i] and all(p.startswith("0.") and len(p) == 8 for p in parts), (i, q)
    rows.append(f"{i},{q}"); print(f"F{i}: {q}    pred {PRED[i]}")
open("Week10_queries.csv", "w").write("function,query_string\n" + "\n".join(rows) + "\n")
