"""Capstone BBO - Module 18 (Week 7) query selection.  16 portal rounds of data.

Round-6 outcome per function:
  F1  analytic centre  -> 1.99999997.  The isotropic-Gaussian closure was right; max = 2.
  F2  x2 = 0.729728    -> 0.587899 << 0.650310.  Separable-quadratic model FALSIFIED.
  F3  global UCB corner-> -0.075096 << -0.025801.  x1/x2 are not inactive away from the peak.
  F4  x3 = 0.383817    -> 0.599274 < 0.680110.  x3 now bracketed on an equal-spaced triple.
  F5  x1 = 0.52        -> 4717.92 NEW BEST.  The ln-quadratic vertex at 0.4245 was wrong.
  F6  5-coord UCB move -> -0.439094 NEW BEST (+0.142), the largest single gain of the round.
  F7  trust-region UCB -> 1.869822 < 1.941004.  First F7 failure; the x5/x6 step overshot.
  F8  d5=0.80, d8=0.20 -> 9.837460 < 9.839469.  dim 8 at 0.20 costs 0.007; revert it.

Surrogate hyperparameters were tuned this round rather than assumed (tune.py): per function,
a grid over kernel family {RBF, Matern-5/2} x lengthscale floor {0.03, 0.06, 0.15, 0.30},
selected on 5-fold held-out RMSE.  Matern won on five of eight functions and the selected floor
ranges from 0.03 to 0.30; best-versus-worst setting differs by up to 1.7x in held-out error.
Selecting on held-out rather than in-sample fit is the discipline the NVIDIA ensemble paper
demonstrates the need for.

Policy this round - four analytic closures, two single-coordinate probes, one tuned-GP UCB
move, one bound test:
  F1  FINISH.  Refit the isotropic Gaussian on all five near-peak readings and query the
      refined centre.  We are ~2.5e-6 from the true centre, below the 1e-6 query resolution,
      so the remaining headroom is 3e-8.  F1 needs no further rounds.
  F2  the three x2 readings now sit on the x1 ridge (0.681098 +- 0.0016), so no correction is
      needed: vertex 0.856190, predicted 0.678307 against an incumbent of 0.650310.  This is
      the same coordinate the falsified model got wrong, tested now without the assumption
      that broke it.
  F3  revert to exploitation.  The round-6 global probe cost 0.049 and confirmed no better
      basin, so take the only move with positive expected value: the x2 vertex at 0.617816.
  F4  x3 vertex 0.366030 (+0.0013, curvature -259) is banked, and the round's information goes
      to x2, frozen at 0.417500 since round 4.  The x3 effect is known exactly from an
      equally spaced triple, so the residual still attributes to x2.
  F5  every coordinate of this function has improved monotonically to its bound, and x1 is the
      last one free: slope rose from 557 to 1566 per unit between the last two points and the
      three-point fit is convex.  Test the corner directly.
  F6  repeat what worked.  The tuned GP ranks the pattern continuation BELOW the incumbent
      (mu -0.52 at a full step) and prefers a fresh UCB point, so take the surrogate over the
      pattern: trust region 0.15, beta 0.6.
  F7  x5 is closed - its vertex 0.305290 reproduces the incumbent to 5e-5 - so the line search
      is finished.  Isolate x4, the one coordinate with a consistent three-round improving
      trend that has never been moved on its own.
  F8  bank the dim-5 closure with dim 8 returned to 0.893085: predicted 9.84487, a safe new
      best.  The GP put +0.14 on moving dim 8 and was wrong by 0.15, so its prediction for the
      other never-varied dimensions is not currently worth a round.
"""
import numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

rng = np.random.default_rng(18)
S = json.load(open('gp_settings.json'))


def load(i):
    return np.load(f'data_w7/f{i}_X.npy'), np.load(f'data_w7/f{i}_y.npy')


def gp_for(i):
    """GP with the kernel family and lengthscale floor chosen by 5-fold CV for this function."""
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

# ---- F1: refit the isotropic Gaussian on all five near-peak readings
P = np.array([[0.628541, 0.628542], [0.62853, 0.629254], [0.6286, 0.6296],
              [0.635252, 0.623823], [0.622, 0.622]])
Y = np.array([1.9999999703867664, 1.997552645210207, 1.994594244405756,
              1.700759293907156, 1.6275243006492137])
A = np.c_[P[:, 0], P[:, 1], -(P[:, 0] ** 2 + P[:, 1] ** 2), np.ones(len(Y))]
s, *_ = np.linalg.lstsq(A, np.log(Y), rcond=None)
Q[1] = np.array([s[0], s[1]]) / (2 * s[2])

# ---- F2: x2 vertex from three readings taken on the x1 ridge
Q[2] = np.array([0.681098, vertex([0.729728, 0.926564, 0.999999],
                                  [0.587899, 0.650310, 0.561396])])

# ---- F3: x2 vertex (the only positive-expected-value move left)
Q[3] = probe(3, {1: vertex([0.55, 0.611593, 0.68], [-0.0376701, -0.0258012, -0.0357643])})

# ---- F4: bank the x3 vertex, spend the information on the frozen x2
Q[4] = probe(4, {2: vertex([0.343807, 0.363817, 0.383817], [0.553209, 0.680110, 0.599274]),
                 1: 0.437500})

# ---- F5: the last free coordinate, tested at its bound
Q[5] = probe(5, {0: 0.999999})

# ---- F6: tuned-GP UCB inside a 0.15 trust region
g6, X6, y6 = gp_for(6)
inc6 = X6[np.argmax(y6)]
lo, hi = np.maximum(0, inc6 - 0.15), np.minimum(1, inc6 + 0.15)
C = rng.uniform(lo, hi, (200000, 5))
mu, sd = g6.predict(C, return_std=True)
Q[6] = np.clip(C[np.argmax(mu + 0.6 * sd)], 1e-6, 1 - 1e-6)

# ---- F7: isolate x4
Q[7] = probe(7, {3: 0.207900})

# ---- F8: bank the dim-5 closure, revert dim 8
Q[8] = probe(8, {4: vertex([0.403935, 0.653935, 0.903935], [9.766436, 9.834202, 9.839469]),
                 7: 0.893085})

print("function,query_string")
for i in range(1, 9):
    print(f"{i}," + "-".join(f"{v:.6f}" for v in Q[i]))
