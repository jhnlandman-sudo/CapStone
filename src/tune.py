"""Select the GP surrogate's hyperparameters per function by 5-fold held-out RMSE.

Grid: kernel family {RBF, Matern-5/2} x lengthscale lower bound {0.03, 0.06, 0.15, 0.30}.
F5 is modelled on log(y).  Writes gp_settings.json to the current directory.
Run from the repository root after build_dataset.py:  python src/tune.py
"""
import json, pathlib, warnings
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(i):
    df = pd.read_csv(ROOT / "data" / "processed" / f"function_{i}.csv")
    return df.filter(regex=r"^x\d+$").to_numpy(), df["y"].to_numpy()


def make(kind, d, lo):
    base = (RBF(np.full(d, .3), (lo, 2.)) if kind == "rbf"
            else Matern(np.full(d, .3), (lo, 2.), nu=2.5))
    return ConstantKernel(1., (1e-3, 1e3)) * base + WhiteKernel(1e-4, (1e-8, 1.))


GRID = [(k, l) for k in ("rbf", "matern") for l in (0.03, 0.06, 0.15, 0.30)]
best = {}
for i in range(1, 9):
    X, y = load(i); z = np.log(y) if i == 5 else y
    rows = []
    for kind, lo in GRID:
        errs = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            g = GaussianProcessRegressor(kernel=make(kind, X.shape[1], lo), normalize_y=True,
                                         n_restarts_optimizer=6, random_state=0).fit(X[tr], z[tr])
            errs.append(np.mean((g.predict(X[te]) - z[te]) ** 2))
        rows.append((float(np.sqrt(np.mean(errs))), kind, lo))
    rows.sort(); r = rows[0]
    best[str(i)] = {"kernel": r[1], "ls_lo": r[2], "cv_rmse": r[0]}
    print(f"F{i}: {r[1]}/{r[2]}  cv_rmse={r[0]:.4g}")
json.dump(best, open("gp_settings.json", "w"), indent=1)
