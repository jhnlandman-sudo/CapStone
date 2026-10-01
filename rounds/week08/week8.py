"""Capstone BBO - Module 19 (Week 8) query selection.  17 portal rounds of data.

Round-7 outcome per function (six new bests):
  F1  refined centre  -> 1.99999999.  SOLVED; the function is 2*exp(-r^2/2s^2), max = 2.
  F2  x2 = 0.856190   -> 0.545329 << 0.650310.  The x2 quadratic is FALSIFIED AGAIN - the wide
      readings hide a dip at 0.856 and a narrow peak at 0.9266, not a single parabola.
  F3  x2 = 0.617816   -> -0.026674 < -0.025801.  The peak sits below 0.611593, not above it.
  F4  x3 vertex + x2 step -> 0.477960.  x3 was worth +0.0013, so the x2 step cost 0.203:
      curvature about -1016, against -855 for x1 and -259 for x3.
  F5  corner (1,1,1,1)-> 8662.41 NEW BEST, +84%.  Monotone-to-bound confirmed for x1 too.
  F6  tuned-GP UCB    -> -0.061133 NEW BEST (+0.378).  Second UCB win in succession.
  F7  x4 = 0.207900   -> 2.047273 NEW BEST (+0.106).  x4 is still paying, 3.54 per unit.
  F8  banked closure  -> 9.844870, matching the predicted 9.84487 to five decimals.

Surrogate hyperparameters re-tuned on the new data (tune.py, 5-fold held-out RMSE).  The
selection moved with the extra point - f1 and f3 both switched kernel family - which is the
argument for re-running the grid every round rather than fixing it once.

Policy this round - four local closures, two probes, one UCB move, one interaction test:
  F1  SOLVED.  Re-fit on all near-peak readings and query the centre; the last two refinements
      returned 1.99999997 then 1.99999999, so the remaining headroom is ~1e-8.
  F2  the quadratic failed twice on widely spaced points, so use only the tight, near-symmetric
      bracket around the peak (0.856190 / 0.926564 / 0.999999): vertex 0.931067, predicted
      0.650691.  x1 goes to its own ridge vertex 0.681098 at the same time; both moves are
      sub-0.002 and the expected gain is small but positive.
  F3  same correction in the other direction: refit on the three tightest x2 readings, giving
      0.600419 and a predicted -0.025188.  F3 is converged to about 1e-3.
  F4  bank the x3 vertex with x2 returned to 0.417500: predicted 0.681381.  This also confirms
      the x2 attribution, since the x3 contribution is known exactly.
  F5  the corner is the incumbent, but x2, x3 and x4 were only ever driven to their bounds at
      x1 = 0.224189, so the interaction has never been tested.  Drop x2 to 0.90 with everything
      else at the bound: a fall confirms the corner and closes F5, a rise means there is more.
  F6  the GP now under-predicts its own incumbent by 0.20, and again ranks the pattern
      continuation below a fresh UCB point.  It has been right twice; take the UCB pick, trust
      region 0.12, beta 0.6, and keep treating mu as a ranking only.
  F7  continue the x4 line with a third, larger step to 0.237900 - this makes the x4 vertex
      computable next round whether the value rises or falls.
  F8  dim 6 has never been varied in the top cluster and the GP puts the largest remaining gain
      there.  The identical prediction for dim 8 was wrong by 0.15 last time, so take a
      moderate step to 0.350000 rather than the GP's preferred 0.06.
"""
import numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

rng = np.random.default_rng(19)
S = json.load(open('gp_settings.json'))


def load(i):
    return np.load(f'data_w8/f{i}_X.npy'), np.load(f'data_w8/f{i}_y.npy')


def gp_for(i):
    X, y = load(i)
    z = np.log(y) if i == 5 else y
    d, s = X.shape[1], S[str(i)]
    base = (RBF(np.full(d, 0.3), (s['ls_lo'], 2.0)) if s['kernel'] == 'rbf'
            else Matern(np.full(d, 0.3), (s['ls_lo'], 2.0), nu=2.5))
    k = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-4, (1e-8, 1e0))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=25, random_state=0).fit(X, z), X, y


def vertex(xs, ys):
    a, b, c = np.linalg.solve(np.c_[np.array(xs) ** 2, xs, np.ones(3)], ys)
    return -b / (2 * a)


def probe(i, changes):
    X, y = load(i)
    x = X[np.argmax(y)].copy()
    for d, v in changes.items():
        x[d] = v
    return np.clip(x, 1e-6, 1 - 1e-6)


Q = {}

# ---- F1: refit the isotropic Gaussian on every near-peak reading
P = np.array([[0.628541, 0.628541], [0.628541, 0.628542], [0.62853, 0.629254],
              [0.6286, 0.6296], [0.635252, 0.623823], [0.622, 0.622]])
Y = np.array([1.9999999865499047, 1.9999999703867664, 1.997552645210207,
              1.994594244405756, 1.700759293907156, 1.6275243006492137])
A = np.c_[P[:, 0], P[:, 1], -(P[:, 0] ** 2 + P[:, 1] ** 2), np.ones(len(Y))]
s, *_ = np.linalg.lstsq(A, np.log(Y), rcond=None)
Q[1] = np.array([s[0], s[1]]) / (2 * s[2])

# ---- F2: tight bracket around the peak, x1 on its own ridge vertex
Q[2] = np.array([0.681098, vertex([0.856190, 0.926564, 0.999999],
                                  [0.545329, 0.650310, 0.561396])])

# ---- F3: refit on the three tightest x2 readings
Q[3] = probe(3, {1: vertex([0.55, 0.611593, 0.617816],
                           [-0.0376701, -0.0258012, -0.02667425360297729])})

# ---- F4: bank the x3 vertex, x2 back to its incumbent value
Q[4] = probe(4, {2: vertex([0.343807, 0.363817, 0.383817], [0.553209, 0.680110, 0.599274]),
                 1: 0.417500})

# ---- F5: test the x1-x2 interaction at the corner
Q[5] = probe(5, {1: 0.900000})

# ---- F6: tuned-GP UCB, trust region 0.12
g6, X6, y6 = gp_for(6)
inc6 = X6[np.argmax(y6)]
lo, hi = np.maximum(0, inc6 - 0.12), np.minimum(1, inc6 + 0.12)
C = rng.uniform(lo, hi, (200000, 5))
mu, sd = g6.predict(C, return_std=True)
Q[6] = np.clip(C[np.argmax(mu + 0.6 * sd)], 1e-6, 1 - 1e-6)

# ---- F7: third point on the x4 line
Q[7] = probe(7, {3: 0.237900})

# ---- F8: moderate step on dim 6, the last never-varied coordinate
Q[8] = probe(8, {5: 0.350000})

print("function,query_string")
for i in range(1, 9):
    print(f"{i}," + "-".join(f"{v:.6f}" for v in Q[i]))
