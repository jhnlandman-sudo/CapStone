# Model Card — classify-then-query Bayesian optimisation

*Johan Landman · Imperial College PCMLAI black-box optimisation challenge · final version (round 12 results,
round 13 submitted). Structure follows the programme's Module 25 model card template, informed by the
[Google model card examples](https://modelcards.withgoogle.com/model-reports). The earlier, longer version is kept in
[archive/model_card_v1_round10.md](archive/model_card_v1_round10.md). Per-function detail is in the
[datasheet](datasheet.md).*

## Model Description

**Input:** the full evaluation history of one function: every input vector *x* in [0, 1]^d (d = 2 to 8)
with its output *y*, from the supplied initial design plus all earlier portal rounds.

**Output:** one query per function per round, written in the portal format `0.dddddd-0.dddddd-…`, together
with a written numerical prediction of the result. Both are produced by a seeded script
(`rounds/weekNN/weekNN.py`) and checked by `src/validate_query.py`.

**Model Architecture:** a sequential, model-based optimiser with three layers.

1. **Surrogate: Gaussian process (GP)** per function, scikit-learn `GaussianProcessRegressor`.
   - Kernel: constant × (RBF or Matérn-5/2, one lengthscale per input) + white noise.
   - Kernel family and lengthscale floor chosen per function by 5-fold cross-validation (`src/tune.py`),
     re-run every round from round 7.
   - Outputs normalised; F5 modelled on log *y*.
   - The fitted lengthscales give each input a relevance score, used to decide which coordinates to move.
2. **Acquisition rules**, one chosen per function:
   - Upper confidence bound (UCB, β 0.5–0.8), first globally, then within a trust region;
   - GP posterior-mean maximiser within a small radius;
   - single-coordinate probes, closed with a three-point quadratic vertex;
   - closed-form fits where the structure is known (an isotropic Gaussian solved F1);
   - exact repeats to measure noise.
3. **Policy layer: classify, then query.** Each round, each function is put in one class, and the class
   picks the move:

| Class | Functions (final) | Move |
|---|---|---|
| Solved | F1, F5 | Resubmit the known optimum |
| Noisy | F2, F3, F6 | Repeat the best point; decide on averages |
| Still improving | F7 | One coordinate at a time along the relevant inputs (small GP step where reliable) |
| Converged | F4, F8 | One coordinate at a time, so each result is attributable |

In the final round there is nothing left to learn for, so the rule is pure exploitation: the highest
reading on noise-free functions, and the highest mean minus standard error on noisy ones.

## Performance

**How performance is measured.** All measures use the evaluation history in `data/` (the supplied design
plus 12 scored portal rounds):
1. **Best value found**, the competition's measure, compared with the best point in the supplied data.
2. **Prediction accuracy:** the written prediction for each query against its result.
3. **Surrogate cross-validated RMSE**, used only for model selection.

![Best value found by round](images/best_so_far.png)

| Fn | d | Best in supplied data | Best found | Change |
|---|---|---|---|---|
| F1 | 2 | ≈ 0 | **2.000** | Solved (analytic maximum) |
| F2 | 2 | 0.611 | **0.726** | +0.115 (replicated mean 0.608) |
| F3 | 3 | −0.0348 | **−0.0250** | +0.010 |
| F4 | 4 | −4.03 | **0.744** | +4.77 |
| F5 | 4 | 1,089 | **8,662** | ×8.0 (solved, corner optimum) |
| F6 | 5 | −0.714 | **−0.061** | +0.653 |
| F7 | 6 | 1.365 | **2.390** | +1.025 |
| F8 | 8 | 9.598 | **9.930** | +0.332 |

**Prediction accuracy (rounds 11 and 12).** Single-coordinate moves on noise-free functions landed close
to prediction:
- F4: x4 vertex predicted 0.7431, returned 0.7440;
- F8: within 0.003 in both rounds;
- F7: x2 move predicted 2.3255, returned 2.390.

Multi-coordinate moves and curve fits on noisy functions missed badly:
- F7 six-coordinate step: predicted 2.447, returned 2.202;
- F2 ridge vertex: predicted 0.614, returned 0.463.

The full table is in section 5 of the notebook.

## Limitations

- **Local, not global.** Most late queries lie within 0.1 of the incumbent. "Converged" means converged
  in the basin found; F6 and F8 in particular have large unexplored regions.
- **Smoothness assumption.** The GP assumes one smoothness level everywhere. It over-smooths sharp peaks
  (F4) and cannot represent boundary optima well without help (F5's corner was found by a hypothesis
  test, not by the GP).
- **Weak level predictions away from data.** The GP is good at ranking candidates and poor at predicting
  values where it has little data. Six- and eight-dimensional GP steps missed by up to five standard
  deviations.
- **Noise.** On F2 (noise sd about 0.1) the noise is as large as any local signal, so the method cannot
  improve on it with the budget available.
- **Human policy layer.** Choosing each function's class and move is a documented judgement. The numbers
  are reproducible from seeded code, but another analyst could choose differently.
- **Small budget.** There are 12 scored points per function, so every conclusion rests on very few
  observations.

## Trade-offs

- **Exploration versus exploitation.** Early UCB exploration found the right regions (F4 −4.03 → 0.553 in
  round 1). Later, shrinking trust regions and single-coordinate moves gave reliable but diminishing
  gains (F7: +0.106, +0.075, +0.046). More exploration might have found better basins on F6 or F8, at the
  cost of fewer refinements.
- **Repeats versus new points.** A repeat cannot raise the record, but it reclassified three of the eight
  functions as noisy and stopped several queries being wasted on curve fits to noise.
- **Single-coordinate versus joint moves.** One-coordinate moves are slower in high dimensions but
  attributable and reliable. Joint GP moves can gain more per query (F7 +0.133 in round 10) but failed twice
  when the model extrapolated.
- **Transparency versus automation.** A hand-written policy layer with written predictions is easy to audit
  but needs judgement each round. A fully automated optimiser such as BoTorch would be faster to run but
  harder to explain, and would not have flagged the noise problem as clearly.
- **Where it performs poorly:** noisy functions with small signals (F2, F3); very narrow peaks the kernel
  smooths away (F4); optima on the boundary (F5) unless tested explicitly; and high dimensions with thin
  coverage (F8).
