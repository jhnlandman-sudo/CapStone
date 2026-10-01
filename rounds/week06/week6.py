"""Capstone BBO - Module 17 (Week 6) query selection.  15 portal rounds of data.

Week-5 outcome per function (what the last query taught):
  F1  triangulated centre returned 1.997553 (new best) -> the bump is Gaussian; close it.
  F2  x2 pushed to the bound returned 0.5614 << 0.6503 -> x2 has an interior optimum.
  F3  x2 lowered to 0.550 returned -0.03767 -> x2 now bracketed; vertex 0.6178, gain 1e-4.
  F4  x1 raised to 0.445 returned 0.3204 << 0.6801 -> x1 is sharply peaked at 0.4160.
  F5  x1 raised to 0.4195 returned 4560.5 (new best) -> ln-quad vertex refits to 0.4245.
  F6  x1 raised to 0.450 returned -0.7552 -> x1 bracketed; vertex 0.36915, gain 4e-4.
  F7  GP/UCB returned 1.9410 (new best) -> the surrogate stays well calibrated.
  F8  dim 5 raised to 0.9039 returned 9.8395 -> dim 5 bracketed; vertex 0.8000, gain 5e-3.

Policy this round - three analytic closures, two single-coordinate probes,
two GP/UCB steps, one deliberate global exploration:
  F1  CLOSE ANALYTICALLY.  Four near-peak readings fit ln y exactly (residual 0 to 5 dp)
      as an isotropic Gaussian: centre (0.628541, 0.628542), sigma 0.014410,
      amplitude 1.9999907.  Query the centre; F1 needs no further rounds.
  F2  x1 sits on its exact 3-point vertex 0.681098 (curvature -84.7 - a very narrow
      ridge).  Correcting the four x2 readings onto that ridge gives 0.559 / 0.614 /
      0.651 / 0.561 at x2 = 0.124 / 0.500 / 0.927 / 1.000, whose vertex is 0.7297.
      Test it: confirmed it is worth +15%, falsified it kills the separable model.
  F3  EXPLORE.  The local neighbourhood is closed - the x2 vertex gains 1e-4, x1 and x2
      are flat across the whole box (ARD lengthscale at the 2.0 ceiling) and x3 sits at
      its local mean-optimum.  Global UCB (beta 0.5) leaves the incumbent altogether.
  F4  x3 is the only coordinate still paying.  0.343807 -> 0.553, 0.363817 -> 0.680.
      A third equally spaced point at 0.383817 makes the x3 vertex exact next round.
  F5  x1 at the corner: the ln-quad vertex refits to 0.4245, i.e. the incumbent, worth
      +0.07 out of 4560.  Spend the round bracketing instead - 0.52 turns two points into
      three and closes F5 for good.  Best-so-far is retained, so the probe is free.
  F6  x1 closed at its vertex 0.369150; GP/UCB over the remaining four coordinates inside
      a 0.15 trust region.  The GP's unconstrained optimum lies outside the region in all
      four, so the query sits on the trust-region face - a deliberately capped step,
      because the GP under-fits the incumbent by 0.085 (mu -0.666 vs -0.581 observed).
  F7  GP/UCB, beta 0.6, trust region 0.09.  Checked the two coordinates frozen since
      week 3 (x2, x3): both genuinely flat, gain < 0.002.  No freeze artefact here.
  F8  dim 5 closed at its vertex 0.800003 (+0.005).  The round's information goes to
      dim 8, frozen at 0.893085 since week 1 and never tested - the GP puts +0.14 on
      moving it to 0.20, the largest predicted gain anywhere in F8.  The dim-5 effect is
      known to +-0.001, so the residual still attributes cleanly to dim 8.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

rng = np.random.default_rng(17)


def load(i):
    return np.load(f'data_w6/f{i}_X.npy'), np.load(f'data_w6/f{i}_y.npy')


def fit_gp(X, z, lo=0.06, hi=2.0):
    d = X.shape[1]
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.full(d, 0.3), (lo, hi))
         + WhiteKernel(1e-4, (1e-8, 1e0)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=20, random_state=0).fit(X, z)


def ucb(gp, inc, free, trust, beta, n=80000):
    lo = np.maximum(0.0, inc - trust); hi = np.minimum(1.0, inc + trust)
    C = np.tile(inc, (n, 1)); C[:, free] = rng.uniform(lo[free], hi[free], (n, free.sum()))
    mu, sd = gp.predict(C, return_std=True)
    top = C[np.argsort(-(mu + beta * sd))[:10]]

    def neg(xf):
        x = inc.copy(); x[free] = xf
        m, s = gp.predict(x.reshape(1, -1), return_std=True)
        return -(m[0] + beta * s[0])

    bx, bv = None, np.inf
    for x0 in top:
        r = minimize(neg, x0[free], bounds=list(zip(lo[free], hi[free])), method='L-BFGS-B')
        if r.fun < bv:
            bv = r.fun; x = inc.copy(); x[free] = r.x; bx = x
    return np.clip(bx, 1e-6, 1 - 1e-6)


def probe(i, dim, val):
    X, y = load(i); x = X[np.argmax(y)].copy(); x[dim] = val
    return np.clip(x, 1e-6, 1 - 1e-6)


def vertex(xs, ys):
    a, b, c = np.linalg.solve(np.c_[np.array(xs) ** 2, xs, np.ones(3)], ys)
    return -b / (2 * a), a


Q = {}

# ---- F1: exact isotropic-Gaussian closure on the four near-peak readings
P = np.array([[0.62853, 0.629254], [0.6286, 0.6296], [0.635252, 0.623823], [0.622, 0.622]])
Y = np.array([1.997552645210207, 1.994594244405756, 1.700759293907156, 1.6275243006492137])
A = np.c_[P[:, 0], P[:, 1], -(P[:, 0] ** 2 + P[:, 1] ** 2), np.ones(4)]
s, *_ = np.linalg.lstsq(A, np.log(Y), rcond=None)
Q[1] = np.array([s[0], s[1]]) / (2 * s[2])

# ---- F2: x1 on its exact vertex, x2 at the ridge-corrected quadratic vertex
x1v, a1 = vertex([0.682637, 0.702637, 0.739667], [0.65031, 0.611205, 0.359876])
pts = [(0.5, 0.703, 0.573418), (0.926564, 0.682637, 0.65031), (0.999999, 0.681098, 0.561396)]
x2v, _ = vertex([p[0] for p in pts], [p[2] - a1 * (p[1] - x1v) ** 2 for p in pts])
Q[2] = np.array([x1v, x2v])

# ---- F3: global UCB, beta 0.5 (deliberate exploration)
X, y = load(3); gp3 = fit_gp(X, y)
C = np.random.default_rng(3).uniform(0, 1, (200000, 3))
mu, sd = gp3.predict(C, return_std=True)
r = minimize(lambda x: -(lambda ms: ms[0][0] + 0.5 * ms[1][0])(
    gp3.predict(x.reshape(1, -1), return_std=True)),
    C[np.argmax(mu + 0.5 * sd)], bounds=[(1e-6, 1 - 1e-6)] * 3, method='L-BFGS-B')
Q[3] = np.clip(r.x, 1e-6, 1 - 1e-6)

Q[4] = probe(4, 2, 0.383817)      # third, equally spaced point on the x3 line
Q[5] = probe(5, 0, 0.520000)      # bracket x1 at the corner

# ---- F6: x1 closed analytically, UCB over the rest inside a 0.15 trust region
X, y = load(6); inc = X[np.argmax(y)].copy()
inc[0], _ = vertex([0.3, 0.3728, 0.45], [-0.70831, -0.581019, -0.755155])
Q[6] = ucb(fit_gp(X, y), inc, np.array([False, True, True, True, True]), 0.15, 0.8)

# ---- F7: GP/UCB on the active coordinates
X, y = load(7); gp7 = fit_gp(X, y); inc = X[np.argmax(y)]
ls = np.atleast_1d(gp7.kernel_.k1.k2.length_scale)
Q[7] = ucb(gp7, inc, ls < 1.9, 0.09, 0.6)

# ---- F8: dim 5 closed analytically, dim 8 unfrozen
v5, _ = vertex([0.403935, 0.653935, 0.903935], [9.766436, 9.834202, 9.839469])
q8 = probe(8, 4, v5); q8[7] = 0.200000
Q[8] = q8

print("function,query_string")
for i in range(1, 9):
    print(f"{i}," + "-".join(f"{v:.6f}" for v in Q[i]))
