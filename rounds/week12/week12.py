"""Capstone BBO - Module 23 (Week 12) query selection.  21 portal rounds of data.

Round-11 outcome (written prediction -> actual):
  F1  centre                  2.0                -> 2.0           noise-free, solved.
  F2  x1 ridge vertex 0.691   0.614              -> 0.463161      FALSIFIED.  The ridge is far narrower than a
                                                                   quadratic across six noisy readings implied.
  F3  GP mean-max r0.05       -0.0335 +/- 0.0095 -> -0.025038     NEW BEST (+0.9 sd above prediction).
  F4  x4 vertex 0.416634      0.7431             -> 0.744037      NEW BEST, prediction within 0.001.
  F5  corner                  8662.405           -> 8662.405001   noise-free, solved.
  F6  x2 vertex 0.341626      -0.048 +/- 0.03    -> -0.146622     FALSIFIED: x2 below 0.365 hurts.
  F7  GP mean-max r0.08       2.4471 +/- 0.0509  -> 2.202056      FALSIFIED (-4.8 sd).  The six-coordinate step overshot.
  F8  dim4 -> 0.138786        9.9285 +/- 0.0038  -> 9.925247      NEW BEST (+0.012), close to prediction.

Policy (round 12 of 13 - last round that can still afford a probe):
  F1, F5  resubmit.
  F2  back to the best-mean point (three readings, mean 0.658) - replicate #4.
  F3  GP posterior-mean maximiser, tighter radius 0.03 around the new best.
  F4  every coordinate tried upward has hurt (x1 +0.03, x2 +0.02); probe x2 DOWN by 0.005.
  F6  three x2 readings now bracket the peak: vertex 0.368189 ~ incumbent, i.e. x2 is closed; query it.
  F7  joint 6-D steps have now failed as well as succeeded - revert to ONE coordinate:
      GP line scan puts the only material gain on x2 (0.302 -> 0.262, +0.024).
  F8  dim 4 is now flat; the GP's largest single-coordinate gain is dim 8 (+0.008). Take a
      moderate -0.10 step (the -0.69 step in round 6 failed).
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
S = json.load(open('gp_settings.json'))
rng = np.random.default_rng(23)

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
def gp_pred(i, x):
    g, *_ = gp_for(i); m, s = g.predict(x[None], return_std=True); return f"GP {m[0]:.4f} +/- {s[0]:.4f}"

Q, P = {}, {}
Q[1] = inc(1); P[1] = "2.0"
Q[5] = inc(5); P[5] = "8662.405"
Q[2] = np.array([0.682637, 0.926564]); P[2] = "0.66 +/- 0.06 (mean of three readings)"
g3, X3, y3 = gp_for(3); x3 = inc(3)
C = rng.uniform(np.maximum(0, x3 - .03), np.minimum(1, x3 + .03), (200000, 3))
mu, sd = g3.predict(C, return_std=True); j = np.argmax(mu)
Q[3] = np.clip(C[j], 1e-6, 1 - 1e-6); P[3] = f"GP {mu[j]:.4f} +/- {sd[j]:.4f}"
Q[4] = probe(4, {1: 0.4175 - 0.005}); P[4] = "0.73 +/- 0.02 (cost <= 0.013 if x2 is at its peak)"
c = np.polyfit([0.341626, 0.365418, 0.4141], [-0.146622, -0.074, -0.292534], 2)
v6 = -c[1] / (2 * c[0]); Q[6] = probe(6, {1: v6}); P[6] = f"{np.polyval(c, v6):.3f} (quadratic) ; noise ~0.02"
Q[7] = probe(7, {1: 0.261928}); P[7] = gp_pred(7, Q[7])
Q[8] = probe(8, {7: 0.893085 - 0.10}); P[8] = gp_pred(8, Q[8])

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
rows = []
for i in range(1, 9):
    q = "-".join(f"{v:.6f}" for v in Q[i]); parts = q.split("-")
    assert len(parts) == DIMS[i] and all(p.startswith("0.") and len(p) == 8 for p in parts), (i, q)
    rows.append(f"{i},{q}"); print(f"F{i}: {q}    pred {P[i]}")
open("Week12_queries.csv", "w").write("function,query_string\n" + "\n".join(rows) + "\n")
