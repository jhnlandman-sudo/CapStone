# Model card — "classify-then-query" Bayesian optimisation for the BBO capstone

*Structure follows Mitchell et al., "Model Cards for Model Reporting" (2019), as adapted in PCMLAI
Mini-lesson 21.2. The data are described in [datasheet.md](datasheet.md).*

---

## 1. Overview

| | |
|---|---|
| **Name** | Classify-then-query GP optimisation |
| **Type** | Sequential, model-based black-box optimiser. A Gaussian-process (GP) surrogate with an upper-confidence-bound (UCB) or posterior-mean acquisition, combined with analytic local closures (quadratic and Gaussian fits) and deliberate replicate queries. |
| **Version** | v10, the round-10 submission (29 September 2026). Each round's exact code is kept in `rounds/weekNN/`. |
| **Owner** | Johan Landman |
| **Libraries** | Python 3; scikit-learn 1.8.0 (`GaussianProcessRegressor`, `RBF`/`Matern` + `WhiteKernel`); NumPy; SciPy; pandas. |
| **Inputs → output** | The full evaluation history of one function → one query in [0, 1]^d, written as `0.dddddd` strings, plus a written numerical prediction. |

## 2. Intended use

**Suitable for:**
- Maximising expensive, low-to-moderate-dimensional functions (d ≤ 8 here), where each
  evaluation is costly and the budget is tens of queries, not thousands.
- Settings where every decision must be auditable: each query is produced by seeded code and
  carries a prediction that the next result confirms or falsifies.

**Avoid for:**
- **Problems with large evaluation budgets or cheap evaluations.** Population methods (for
  example CMA-ES) or plain random search are simpler there, and GP fitting scales as n³.
- **High dimensions (d ≳ 20) without structural assumptions.** The GP's coverage in this
  project is already thin at d = 8.
- **Functions with discontinuities, categorical inputs or non-stationary behaviour.** The kernel
  assumes one smoothness everywhere.
- **Any setting that needs a global-optimality guarantee.** The method converges to the basin it
  finds.
- **Automated use without review.** A human policy layer chooses the move type for each
  function (see §5).

## 3. Details — strategy and how it evolved over ten rounds

1. **Rounds 1–3: GP-UCB and directional moves.** A GP with a separate lengthscale per input
   dimension (automatic relevance determination, ARD, RBF kernel) plus a white-noise term was
   fitted to the supplied initial design. It was used alongside simple directional moves away
   from the best initial points. The fitted lengthscales were the main diagnostic of which
   inputs matter.
2. **Rounds 4–6: local structure.** Once incumbents emerged, I switched to cheaper and more
   exact tools:
   - **Single-coordinate probes and three-point quadratic vertices**, for example F4 x3,
     F6 x1 and F8 dim 5.
   - **A closed-form isotropic-Gaussian fit that solved F1.** Its centre is (0.628541,
     0.628541) and its amplitude is 2.
   - **Trust-region UCB for F6 and F7**, with radius 0.09–0.15 and β 0.5–0.8.

   I also added a standing *freeze-artefact* check: a coordinate never varied looks inactive to
   the GP, so its whole range is scanned before accepting that. This released F8's dimension 8
   for testing, although the test then showed the GP's predicted gain was wrong.
3. **Round 7: tuned surrogate.** `src/tune.py` selects the kernel family (RBF or Matérn-5/2) and
   the lengthscale floor per function by 5-fold cross-validated root-mean-square error (RMSE),
   and is re-run every round. Since round 6, every query has had a written numerical
   prediction.
4. **Rounds 8–9: noise recognised.** Three vertex predictions on F2 and F3 kept failing. With
   18 points, the re-tuned GP fitted a real noise term (sd 0.036 on F2, 0.11 on F6). Exact
   replicates then settled it: F5 returned 8662.405 twice, so it is noise-free; F2 returned
   0.650 and then 0.726 at the same input, so it is noisy. For multi-coordinate moves I adopted
   half-steps after an eight-dimensional GP move on F8 failed.
5. **Round 10: classify, then query.** Each function now gets the policy its evidence supports:

| Class | Functions | Rule |
|---|---|---|
| Solved / at a bound | F1, F5 | Resubmit the known optimum |
| Noisy | F2, F3, F6 | Replicate the best point; decide on averages, not single draws |
| Noise-free, still improving | F7 | GP posterior-mean maximiser within a trust region (radius 0.08) |
| Noise-free, converged | F4, F8 | One coordinate at a time, so the effect can be attributed |

## 4. Performance

**Metrics.**
1. **Best observed *y*.** This is the competition's measure.
2. **Improvement over the best point in the supplied initial design.** This isolates what the
   queries added.
3. **Rounds producing a new best.** This shows the rate of progress.
4. **Prediction accuracy.** Written prediction against actual result, which checks the
   surrogate's calibration.
5. **Surrogate cross-validated RMSE,** used for model selection only.

| Fn | d | Best initial | Best to date | Change | New-best rounds | Status (after round 9) |
|---|---|---|---|---|---|---|
| F1 | 2 | ≈ 0 | **2.000000** | +2.0 | 1, 5, 6, 7 | Solved, analytic maximum; noise-free |
| F2 | 2 | 0.6112 | **0.7257** | +0.114 | 4, 9 | Noisy (sd ≈ 0.05); best point read twice |
| F3 | 3 | −0.0348 | **−0.0258** | +0.009 | 1 | Converged locally; noisy (sd ≈ 0.009) |
| F4 | 4 | −4.0255 | **0.6801** | +4.71 | 1, 4 | Converged; very sharp peak (x4 curvature ≈ −2700) |
| F5 | 4 | 1,088.9 | **8,662.4** | ×8.0 | 1–7 | Solved, optimum at the corner (1,1,1,1); noise-free |
| F6 | 5 | −0.7143 | **−0.0611** | +0.653 | 1, 6, 7 | Uncertain; best point being replicated |
| F7 | 6 | 1.3650 | **2.1681** | +0.803 | 2, 4, 5, 7, 8, 9 | Still improving, with diminishing steps |
| F8 | 8 | 9.5985 | **9.9130** | +0.315 | 1, 3–5, 7, 8 | Converged; best remaining single-coordinate gain ≈ +0.005 |

**Prediction accuracy.**
- **Round 9:** four of eight predictions held within their stated bands (F1, F5, F6 and F7; F7
  came in at 2.1681 against a predicted 2.1674 ± 0.014). Four were falsified (F2, F3, F4, F8).
- **Round 7:** F8 matched its prediction to five decimals, while F2 and F3 were falsified.
- **Pattern:** the GP's predictions are reliable along single lines on noise-free functions
  (F7, F8 dim 6), and unreliable for multi-coordinate moves and on noisy functions.

## 5. Assumptions, constraints and failure modes

**Assumptions.**
- **Stationarity and smoothness.** The functions are assumed smooth, with one lengthscale per
  dimension everywhere and a single Gaussian noise level everywhere.
- **Local additivity.** The single-coordinate closures assume the function is locally additive
  in its coordinates.
- **Best-so-far scoring.** A query that scores badly costs information, not score.

**Constraints.**
- **Budget:** one query per function per round, about 13 rounds in total.
- **Output format:** six-decimal values in [0, 1).
- **Scale:** GP fitting is cubic in n, which is trivial at n ≤ 49 but would matter at scale.

**Failure modes seen in this project.**
1. **Fitting quadratics to noise.** Three F2/F3 vertex predictions failed before the noise was
   measured.
2. **Over-smoothing narrow peaks.** Cross-validation prefers long lengthscales on F4, so the GP
   under-rates its sharp peak. The GP also predicts 0.61 at an F2 point that has twice returned
   ≥ 0.65.
3. **The GP as a level estimate.** The GP misses its own incumbent by 0.2 on F6, so it is used
   to rank candidates, not to predict values.
4. **Multi-dimensional steps.** The F8 dim-8 and eight-dimension half-step moves both missed.
5. **Local convergence mistaken for global.** 57% of queries sit near incumbents, so
   "converged" means converged in the basin found.
6. **Human policy layer.** The move type for each function is a documented judgement, not an
   algorithm. The numbers can be reproduced, but the policy choices are subjective.

## 6. Ethical considerations — transparency, reproducibility and adaptation

**Reproducibility.** The functions are synthetic, so there is no direct social harm; the ethical
stake is honesty about what the results show.
- Raw data are kept read-only, and every processed file is regenerated by code.
- Every query comes from seeded code and passes a format validator.
- Library versions are pinned in `requirements.txt`.

Together, these mean a reviewer can regenerate every submitted string.

**Honest reporting.** The prediction-versus-actual record in each round's docstring means
failures are reported rather than quietly dropped. That record is what exposed the noise on F2
and F3.

**Adaptation to real problems.** The same discipline carries over to real expensive optimisation
problems, such as tuning an engineering design or an actuarial model's parameters:
- state the assumption behind each move;
- replicate before trusting a single reading;
- report the explored region honestly, so no one mistakes "best found" for "best possible".

The stated failure modes (noise, over-smoothing, local convergence) tell a future user where
the method needs changing before it is reused.

## 7. Is more detail needed?

The current structure is sufficient for a reviewer to follow and reproduce the work.
- **Rationale:** the card gives the rationale and the outcomes.
- **Per-round decisions:** these live in each round's script docstring, which gives claim,
  evidence and prediction for every function.
- **Data:** the datasheet covers these.

Adding per-query detail here would duplicate those files and bury the key points. The one
addition that *would* improve it is a machine-readable log (`rounds/predictions.csv`, with
columns function, round, prediction, interval and actual), so prediction accuracy can be
computed rather than read. That is planned for the next round.

## References
See [references.md](references.md). Also cited here: Gebru, T. et al. (2021), *Datasheets for
Datasets*, Communications of the ACM 64(12); Mitchell, M. et al. (2019), *Model Cards for Model
Reporting*, Proceedings of FAT* 2019.
