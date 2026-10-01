"""Capstone BBO - Module 20 (Week 9) query selection.  18 portal rounds of data.

Round-8 outcome (prediction -> actual):
  F1  centre            2.0        -> 2.0 (identical to round 7 to 1e-16: F1 is noise-free). SOLVED.
  F2  (0.681098,0.931)  0.650691   -> 0.582926  FALSIFIED a third time.
  F3  x2 = 0.600419     -0.025188  -> -0.031254 FALSIFIED.
  F4  x3 vertex banked  0.681381   -> 0.675667  short by 0.006; x3 is closed at 0.363817.
  F5  x2 = 0.90         fall       -> 7073 < 8662. Corner confirmed; F5 closed.
  F6  tuned-GP UCB      > -0.061   -> -0.292534 first UCB miss after two wins.
  F7  x4 = 0.237900     rise       -> 2.122401 NEW BEST (+0.075; previous step +0.106).
  F8  dim 6 = 0.35      rise       -> 9.913004 NEW BEST (+0.068).

What changed this round: re-tuned GPs now fit a real noise term on F2 (sd 0.036), F3 (0.008)
and F6 (0.11).  All five F2 readings at x1 = 0.681 lie in 0.545-0.588 whatever x2 is; the lone
0.650 is +1.7 sd above the GP mean at the same point.  The three 'falsified' vertex fits on F2/F3
were fitted to noise.  F1, F7, F8 remain effectively noise-free.

Policy: exploit where gains are steady, measure where noise dominates, never fit a quadratic to
readings closer together than the noise.
  F1  resubmit centre (solved).
  F2  REPLICATE the incumbent exactly.  Same input -> any difference is pure noise.  Pred 0.59.
  F3  GP posterior-mean maximiser in radius 0.05 (not UCB).  Pred -0.027 +/- 0.009.
  F4  x4 is the one coordinate never moved on the incumbent line; +0.005 step, direction from a
      local diagonal quadratic (grad +1.5).  Pred 0.68 +/- 0.03 (cost <= 0.023 if x4 is optimal).
  F5  resubmit the corner: tests whether F5 is noise-free.  Pred 8662.405.
  F6  GP mean-max in radius 0.08, beta 0.3 (was 0.12 / 0.6).  Pred -0.27 (GP) ; incumbent -0.061.
  F7  x4 to 0.28: GP line peak 0.28 (2.167 +/- 0.014); three-point quadratic vertex 0.295 (2.179).
      Gains per 0.03 step 0.106 -> 0.075: diminishing, so stop at the GP peak, not the vertex.
  F8  half-step toward the GP mean-max in radius 0.15: pred 9.932 +/- 0.016 (full step 9.945
      +/- 0.030 moves all eight coordinates; the dim-8 lesson says halve it).
"""
import numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
rng = np.random.default_rng(20)
S = json.load(open('gp_settings.json'))

def load(i): return np.load(f'data_w9/f{i}_X.npy'), np.load(f'data_w9/f{i}_y.npy')
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
def tr_mean_max(i, rad, beta, n=200000):
    g, X, y = gp_for(i); x0 = inc(i)
    C = rng.uniform(np.maximum(0, x0 - rad), np.minimum(1, x0 + rad), (n, len(x0)))
    mu, sd = g.predict(C, return_std=True); j = np.argmax(mu + beta * sd)
    return np.clip(C[j], 1e-6, 1 - 1e-6), mu[j], sd[j]

Q, PRED = {}, {}
Q[1] = inc(1);                               PRED[1] = "2.0"
Q[2] = inc(2);                               PRED[2] = "0.59 (replicate; record 0.650 read as noise)"
Q[3], m, s = tr_mean_max(3, 0.05, 0.0);      PRED[3] = f"{m:.4f} +/- {s:.4f}"
Q[4] = probe(4, {3: 0.428034 + 0.005});      PRED[4] = "0.68 +/- 0.03"
Q[5] = inc(5);                               PRED[5] = "8662.405"
Q[6], m, s = tr_mean_max(6, 0.08, 0.3);      PRED[6] = f"{m:.4f} +/- {s:.4f} (GP); incumbent -0.0611"
Q[7] = probe(7, {3: 0.280000})
g7, *_ = gp_for(7); m, s = g7.predict(Q[7][None], return_std=True); PRED[7] = f"{m[0]:.4f} +/- {s[0]:.4f}"
g8, X8, y8 = gp_for(8); x8 = inc(8)
C = np.random.default_rng(20).uniform(np.maximum(0, x8 - .15), np.minimum(1, x8 + .15), (200000, 8))
mu = g8.predict(C); t8 = C[np.argmax(mu)]
Q[8] = np.clip(x8 + 0.5 * (t8 - x8), 1e-6, 1 - 1e-6)
m, s = g8.predict(Q[8][None], return_std=True); PRED[8] = f"{m[0]:.4f} +/- {s[0]:.4f}"

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
rows = []
for i in range(1, 9):
    q = "-".join(f"{v:.6f}" for v in Q[i])
    parts = q.split("-")
    assert len(parts) == DIMS[i] and all(p.startswith("0.") and len(p) == 8 for p in parts), (i, q)
    rows.append(f"{i},{q}")
    print(f"F{i}: {q}    pred {PRED[i]}")
open("Week9_queries.csv", "w").write("function,query_string\n" + "\n".join(rows) + "\n")
