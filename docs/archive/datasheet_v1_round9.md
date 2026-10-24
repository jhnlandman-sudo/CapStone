# Datasheet — BBO capstone query history

*Format follows Gebru et al., "Datasheets for Datasets" (2021), as adapted in PCMLAI Mini-lesson 21.1.
Version 1.0, 29 September 2026: covers the initial design plus portal rounds 1–9. The round-10
queries have been submitted, but their results are not yet in the data.*

---

## 1. Motivation

**Why the data set exists.** It is the complete record of every evaluation made of eight unknown
("black-box") functions in the Imperial College PCMLAI capstone black-box optimisation (BBO)
challenge. The functions can only be observed by submitting an input and receiving one output.
Each round allows one query per function, so this file is the *only* information anyone has
about them.

**Task it supports.** Sequential maximisation of each function under a very small evaluation
budget. In practice that means:
1. fitting surrogate models, here Gaussian processes, to decide the next query;
2. checking, after each round, the numerical prediction made for each query;
3. reproducing every submitted query from the data plus a fixed-seed script.

**Who created it.** Johan Landman, as a participant in the Imperial College London Professional
Certificate in Machine Learning and Artificial Intelligence. The functions, the initial design
and the portal belong to the programme; the query choices, preprocessing and documentation are
mine. It was produced as coursework, with no external funding.

## 2. Composition

**What an instance is.** One evaluation: an input vector *x* ∈ [0, 1]^d and a real-valued output
*y*. There are no labels in the supervised sense; *y* is the target.

| Function | d | Initial points | Portal queries | Total | Best initial *y* | Best *y* to date |
|---|---|---|---|---|---|---|
| F1 | 2 | 10 | 9 | 19 | ≈ 0 (7.7e-16) | 2.000000 |
| F2 | 2 | 10 | 9 | 19 | 0.611205 | 0.725670 |
| F3 | 3 | 15 | 9 | 24 | −0.034835 | −0.025801 |
| F4 | 4 | 30 | 9 | 39 | −4.025540 | 0.680110 |
| F5 | 4 | 20 | 9 | 29 | 1,088.86 | 8,662.41 |
| F6 | 5 | 20 | 9 | 29 | −0.714265 | −0.061133 |
| F7 | 6 | 30 | 9 | 39 | 1.364970 | 2.168082 |
| F8 | 8 | 40 | 9 | 49 | 9.598480 | 9.913004 |
| **All** | | **175** | **72** | **247** | | |

**Formats.**
- **Raw:** `data/raw/initial/function_<i>/initial_inputs.npy` and `initial_outputs.npy` (NumPy),
  as supplied; `data/raw/portal/inputs.txt` and `outputs.txt`, as downloaded from the portal.
  These have one line per round, eight entries per line, and the inputs are written as Python
  `array([...])` reprs.
- **Processed:** `data/processed/function_<i>.csv`, with columns `source` (initial/portal),
  `round` (0 = initial), `x1…xd`, `y`, plus `summary.csv`.

**Sample or complete?** It is the complete record of my evaluations, but only a very sparse
sample of each function. The output scales differ by four orders of magnitude (F5 reaches
8,662; F3 lives between −0.4 and −0.03), so the functions must never be pooled or compared on
raw *y*.

**Known gaps and structure.**
- **Heavily exploitative sampling.** 41 of the 72 portal queries (57%) lie within 0.1 of their
  function's current best point, and on F4 all nine do. The portal queries describe the
  neighbourhood of each incumbent, not the function.
- **Sparse coverage in high dimensions.** For a random point in F8's cube, the nearest
  observation is typically 0.60 away, against 0.13 in F1–F2. Away from the incumbent basins, the
  initial design is the only information.
- **Bound-seeking.** On F5, 69% of the portal coordinates sit at 0 or 1, because the optimum
  lies at the corner.
- **Deliberate repeats.** F1 (two), F2 (one) and F5 (one) contain exact repeat queries. They are
  kept on purpose, because they are the only direct measurement of noise: F1 and F5 returned
  identical values; F2 returned 0.650 and then 0.726 at the same input.
- **Round 10 pending.** Eight queries have been submitted but have no outputs yet.

**Relationships between instances.** The portal rows are *not* independent draws: each was
chosen using all earlier rows. Any statistic that assumes i.i.d. sampling (a mean *y*, a naive
train/test split) is biased towards the incumbent regions.

**Recommended splits.** None for optimisation. For surrogate model selection I use 5-fold
cross-validation with a fixed seed (`src/tune.py`), and I note that the folds inherit the
clustering bias above.

**Privacy and sensitive content.** None. The data are synthetic numeric function values; there
are no people, personal data or demographic attributes. The portal credentials are not stored
here.

## 3. Collection process

**How queries were generated.** Each round, one query per function was chosen by a Python
script (`rounds/weekNN/weekN.py`) run on the full history to date, then submitted to the portal
as `x1-x2-…-xd` with six decimals. `src/validate_query.py` checks that format before each
submission. The query strategy changed over time (see the model card), in this order:
1. Gaussian-process upper-confidence-bound (GP-UCB) search.
2. Single-coordinate probes and three-point quadratic "closures".
3. Trust-region UCB steps.
4. Cross-validated surrogate tuning, from round 7.
5. Deliberate exact replicates to measure noise, from round 9.

**Deterministic or random?** Deterministic given the data. Where candidates are sampled, the
seed is fixed (`numpy.random.default_rng(<seed>)`, `random_state=0`), so re-running a round's
script reproduces its query strings exactly. The initial design was supplied by the programme,
and its sampling method is not documented.

**Written predictions.** From round 6 onward, each query was stored with a numerical prediction
before submission; the scripts' docstrings record each prediction against the actual result.

**Time frame.** Initial data were received on 3 August 2026, and portal results were collected
weekly from 19 August to 29 September 2026 (rounds 1–9). The programme runs about 13 rounds, so
the data set will grow.

**Ethical review and consent.** Not applicable, since there are no human subjects.

## 4. Preprocessing, cleaning and labelling

**Transformations.**
- The raw portal text is parsed into arrays; each row is checked for the correct dimension and
  for values within [0, 1].
- Nothing is dropped, deduplicated or clipped in the stored data: repeats and outliers are kept.
- F5 is modelled on `log(y)` inside the surrogate only, because its outputs are strictly positive
  and span four orders of magnitude. The stored *y* is untransformed.
- The GP normalises *y* internally (`normalize_y=True`).

**Raw data preserved?** Yes. `data/raw/` is never edited, and `data/processed/` is regenerated
from it by `python src/build_dataset.py`.

## 5. Uses

**Intended uses.**
- Reproducing and auditing my query decisions round by round.
- Fitting and comparing surrogate models on these eight functions.
- Teaching: a small, real record of sequential design, where the effects of exploitation,
  noise and dimension are visible.

**Inappropriate uses.**
- **Estimating global properties of the functions.** The sampling is concentrated around
  incumbents, so the data describe the basins found, not the whole space.
- **Benchmarking optimisers** against other participants' results without accounting for
  different initial designs and query budgets.
- **Treating a single reading on F2, F3 or F6 as exact.** These functions are noisy; the GP's
  estimated noise standard deviation (sd) is 0.055, 0.009 and 0.10 respectively.
- **Any real-world inference.** The functions are synthetic and their generating process is
  unknown.

## 6. Distribution

**Where.** This public GitHub repository, https://github.com/jhnlandman-sudo/CapStone, released
alongside the code that produced every query.

**Terms of use.** The code and documentation are shared for educational and non-commercial use,
with attribution. The underlying functions and initial design were provided by the PCMLAI
programme, so reuse beyond education should respect the programme's terms. There are no fees.

## 7. Maintenance

**Who maintains it.** Johan Landman (repository owner).

**Updates.** After each portal round, the new `inputs.txt` and `outputs.txt` replace the files in
`data/raw/portal/` (the portal file is cumulative). Then `src/build_dataset.py` is re-run, and the
round's script and queries are added under `rounds/`. The version line at the top of this file
records which rounds are included.

**Versioning and errata.** Git history is the version control. Corrections are made only in the
processing code, never by editing the raw files. The dataset will be frozen when the final
round's results are received.
