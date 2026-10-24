# Retrospective on the BBO capstone project (component 25.1)

**Repository:** https://github.com/jhnlandman-sudo/CapStone

**1. Initial codebase**

I built the codebase from scratch on standard public libraries: scikit-learn's
`GaussianProcessRegressor` for the surrogate, with NumPy and pandas for the data handling. I did
not start from a Bayesian-optimisation framework such as BoTorch or scikit-optimize. Three
reasons drove that choice:
- **Control:** each round I wanted to see and change every step myself: the kernel, the noise
  term, the acquisition rule and the candidate search. A framework's default loop would have
  hidden the decisions I most needed to learn from.
- **Fit to the problem:** with one query per function per round and 10–40 starting points, the
  surrogate is the cheap part. The hard part is judgement: when to explore, when to repeat a
  point, when to stop. That needed small, transparent tools rather than a large engine.
- **Reproducibility:** as an actuary I am used to every number being re-creatable and
  auditable. A short script I wrote and can explain line by line meets that standard more easily
  than a framework whose defaults change between versions.

The repository grew into a small pipeline:
- `src/build_dataset.py` rebuilds the full history (the supplied initial points plus every portal
  result) from the raw files each round, with shape and range checks.
- `src/tune.py` selects each function's kernel by 5-fold cross-validation.
- `src/validate_query.py` checks the portal format.
- `rounds/weekNN/` holds one seeded script per round. Its docstring records the previous round's
  predictions against the actual results, and the policy for the new round.

The repository also contains a datasheet, a model card and a references file.

**2. Code modifications, week by week**

**Rounds 1–3: global GP-UCB.** A GP with one lengthscale per input (ARD, RBF kernel) plus a
noise term, maximised with an upper confidence bound (β 0.5–0.8) and simple directional moves
away from the best initial points. This found the promising regions quickly: F4 went from −4.03
in the initial data to 0.553 in round 1, and F5 from 1,089 to 2,917 by round 2. But it was blind
to the shape of each function, and several UCB picks landed at the edges of the box.

**Rounds 4–6: local structure.** Once each function had a clear incumbent, I switched to cheaper
and more exact tools:
- **Single-coordinate probes with three-point quadratic vertices** (F4 x1 and x3, F6 x1, F8 dim 5).
  Moving one coordinate at a time makes each result attributable.
- **A closed-form Gaussian fit for F1.** Four near-peak readings fitted ln y exactly as an
  isotropic Gaussian, giving the centre (0.628541, 0.628541). F1 returned 1.99999997 and then
  exactly 2.0: solved in round 6.
- **Trust-region UCB for F6 and F7.** This kept the GP's search close to the incumbent.

**Round 7: a tuned surrogate.** Rather than fix the GP settings, `tune.py` chose the kernel
family (RBF or Matérn-5/2) and the lengthscale floor for each function on held-out error. Matérn
won on five of eight functions, and the best and worst settings differed by up to 1.7× in error.
From round 6 onwards, every query also carried a written numerical prediction.

**Round 7: the F5 corner.** Each F5 input had improved every time it was pushed upwards, so
instead of another small step I tested the hypothesis that the optimum sits at the corner
(1, 1, 1, 1). F5 went from 4,718 to 8,662 (+84%). A probe at x2 = 0.9 then fell to 7,073, which
confirmed the corner.

**Rounds 8–9: recognising noise.** Three confident vertex predictions on F2 and F3 failed in a
row. The re-tuned GPs began to fit a real noise term, so I tested it directly by resubmitting
identical inputs:
- **F5** returned 8,662.405 exactly, so it is noise-free.
- **F2** returned 0.650 and then 0.726 at the same input, and later 0.598 and 0.460.

I had been fitting curves to noise. From then on, repeats became a deliberate tool, and my rule
became: never fit a quadratic to readings closer together than the noise.

**Rounds 10–12: classify, then query.** Each function was put in a class, and the class chose the
move:
- **Solved (F1, F5):** resubmit.
- **Noisy (F2, F3, F6):** repeat the best point and decide on averages.
- **Still improving (F7):** a GP-guided step in a small trust region.
- **Converged (F4, F8):** one coordinate at a time.

I also ran a relevance analysis from the GP lengthscales, followed by a PCA of my own queries.
Three of F8's eight inputs carried about 70% of the relevance, yet most of my F8 sampling had gone
along two of the least relevant inputs. Joint multi-coordinate GP steps also failed twice:
- **F8:** an eight-dimensional step.
- **F7:** a six-dimensional step, predicted 2.447 and returned 2.202.

After those, I moved back to single coordinates.

**Round 13: pure exploitation.** A rule picks the best-evidenced point. For noise-free functions
that is the highest reading. For noisy functions it is the highest mean minus its standard error,
so one lucky draw cannot beat a replicated point.

**What had the most impact.** Three changes stand out:
- **The F5 corner test:** the largest single gain of the project (+3,944).
- **Moving one coordinate at a time on F7:** it produced new bests of +0.106, +0.075 and +0.046,
   and, after a failed joint step, +0.089 in round 12, to 2.390.
- **Deliberate replicates:** less visible, but just as important. They stopped me spending
   queries on curve fits that could never have worked on F2 and F3.

**3. Final result**

Best point in the supplied data → my best, by function:

- **F1:** 0.000 → my best 2.000 (solved: Gaussian peak)
- **F2:** 0.611 → my best 0.726 (noisy ridge; replicated mean 0.608)
- **F3:** −0.0348 → my best −0.0250 (flat and noisy)
- **F4:** −4.03 → my best 0.744 (very sharp peak)
- **F5:** 1,089 → my best 8,662 (solved: corner)
- **F6:** −0.714 → my best −0.061 (narrow peak, confirmed by a repeat)
- **F7:** 1.365 → my best 2.390 (improving until the end)
- **F8:** 9.598 → my best 9.930 (converged)

**How the score evolved in the final weeks.** It was a story of diminishing returns:
- **Solved early:** F1 and F5 were finished by round 7.
- **Last gains:** in rounds 10–12, the only new bests came from F3 (−0.0250), F4 (0.708 and then
  0.744), F7 (2.301 and then 2.390) and F8 (9.925 and then 9.930).
- **Round 12:** two of its three single-coordinate probes paid (F7 and F8). F4's probe
  (x2 −0.005) fell to 0.739, so the final round reverted to the incumbent.

Round 13 sends every function to its best-evidenced point. Its results were not available at the
time of writing.

**With more time or a fresh start, I would:**
- **Run replicates in round 1.** Two or three repeats of the best initial point for each function
  would have classified each function as noisy or noise-free from the start, and saved the several
  queries I spent fitting curves to F2 and F3's noise.
- **Check relevance before spending queries.** Computing the ARD relevance and checking it against
  my own sampling directions from the start would have stopped me spending F8's budget on inputs
  that barely matter.
- **Keep a machine-readable prediction log.** Rather than predictions in docstrings, a log would
  let me measure the surrogate's calibration across all 13 rounds instead of describing it.
- **Test boundary hypotheses earlier.** F5 rose every time an input was pushed upwards. Testing the
  corner as soon as that pattern appeared would have freed rounds for other functions.

**4. Trade-offs and decisions**

**Exploration versus exploitation.** My balance shifted on purpose as the data grew. In early
rounds, a query's value lay mostly in what it taught, so I used UCB with a moderate exploration
weight. As incumbents emerged, I shrank the trust regions. In the final round, exploration is
worth nothing because no later round can use what it teaches, so every query went to the best
evidence. One feature of the competition shaped this: only the best result so far counts, so a
bad probe costs information, not score. That made cheap, single-coordinate probes worthwhile
until round 12.

**Short-term gain versus long-term learning.** The clearest example was replicates:
- **The cost:** a repeat cannot raise the record in expectation, so it "wastes" a query in the
  round it is spent.
- **The payoff:** F2's four readings (0.650, 0.726, 0.598, 0.460) changed how I treated three of
  the eight functions for the rest of the project. I judged that worth more than several more
  attempts to beat 0.726.

**Model versus direct evidence.** The GP was excellent at ranking candidates and poor at
predicting levels away from its data. It missed by nearly five standard deviations on F7's
six-dimensional step. I resolved this by using the GP to choose a direction, then moving along it
one coordinate at a time, so each result tested one claim.

**Allocating effort across functions.** Every function got one query per round, but my analysis
time did not have to be spread evenly. Once F1 and F5 were solved, I spent almost no time on them
and concentrated on the functions still improving (F4, F7 and F8).

**5. Learning and application**

**The most important lesson: a single good reading is a hypothesis, not a result.** Before
round 9, I treated every output as truth and built ever more careful models on top of it. The
failures on F2 and F3 were not modelling errors in the usual sense. The models were fitting noise
very accurately. Measuring noise directly, by repeating a point, was the cheapest and most
valuable query I made.

This matters because the same trap exists in my work as a life actuary:
- **Expensive runs:** a full projection-model run (FIS Prophet, in my case) can take minutes to
  hours. Calibrating assumptions is exactly a black-box optimisation with expensive evaluations.
- **Run-to-run noise:** stochastic valuations add noise of their own.
- **The workflow I would now use:** establish the noise first, check which inputs actually drive
  the result before searching, plan with a surrogate but confirm with real runs, and record a
  prediction before every run so each result tests something.

The same applies to future competitions. I would spend the first round measuring the problem,
not attacking it.

**What surprised me.** Three things:
- **How low-dimensional the problems were.** F2 is effectively a one-dimensional function in a
   two-dimensional box, F3 is almost entirely x3, and F8's eight inputs reduce to about three
   that matter.
- **That a simple hypothesis beat the surrogate.** The biggest gain came not from the GP but from
   asking "what if the optimum is at the corner?". The surrogate, which extrapolates cautiously,
   would not have proposed that move for several more rounds.
- **How much the written prediction mattered.** Committing to a number before each query turned
   vague disappointments into specific falsifications. Each falsification changed the method,
   not just the next point, and almost every improvement in my process can be traced back to one.
