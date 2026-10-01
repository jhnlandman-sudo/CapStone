"""Rebuild the full evaluation history from the raw files.

Inputs (never edited):
  data/raw/initial/function_<i>/initial_inputs.npy, initial_outputs.npy   - supplied design
  data/raw/portal/inputs.txt, outputs.txt                                   - portal results,
      one line per round, eight entries per line (function 1..8)

Outputs (regenerated every run):
  data/processed/function_<i>.csv  columns: source, round, x1..xd, y
  data/processed/summary.csv       one row per function

Run from the repository root:  python src/build_dataset.py
"""
import re, csv, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


def portal_rounds():
    txt = (RAW / "portal" / "inputs.txt").read_text()
    arrs = [np.array([float(v) for v in re.split(r"[,\s]+", m.strip()) if v])
            for m in re.findall(r"array\(\[(.*?)\]\)", txt, flags=re.S)]
    assert len(arrs) % 8 == 0, len(arrs)
    R = len(arrs) // 8
    ys = [[float(v) for v in l.strip().strip("[]").split(",")]
          for l in (RAW / "portal" / "outputs.txt").read_text().splitlines() if l.strip()]
    assert len(ys) == R, (len(ys), R)
    return R, [arrs[r * 8:(r + 1) * 8] for r in range(R)], ys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, xs, ys = portal_rounds()
    summary = []
    for i in range(1, 9):
        d = DIMS[i]
        X0 = np.load(RAW / "initial" / f"function_{i}" / "initial_inputs.npy")
        y0 = np.load(RAW / "initial" / f"function_{i}" / "initial_outputs.npy").ravel()
        assert X0.shape[1] == d
        rows = [("initial", 0, *x, y) for x, y in zip(X0, y0)]
        for r in range(R):
            x = xs[r][i - 1]
            assert len(x) == d and np.all((x >= 0) & (x <= 1)), (i, r)
            rows.append(("portal", r + 1, *x, ys[r][i - 1]))
        with open(OUT / f"function_{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "round"] + [f"x{k+1}" for k in range(d)] + ["y"])
            w.writerows(rows)
        y = np.array([r[-1] for r in rows]); j = int(np.argmax(y))
        summary.append([i, d, len(y0), R, len(y), f"{y0.max():.6g}", f"{y.max():.6g}",
                        rows[j][1], "-".join(f"{v:.6f}" for v in rows[j][2:-1])])
        print(f"F{i}: d={d} initial={len(y0)} portal={R} best={y.max():.6g} (round {rows[j][1]})")
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "dim", "n_initial", "n_portal", "n_total",
                    "best_initial_y", "best_y", "best_round", "best_x"])
        w.writerows(summary)


if __name__ == "__main__":
    main()
