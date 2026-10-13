# Component 23.1 — reflection on round 12 (21 data points)

**How the strategy has matured.** Early rounds were one GP-UCB pick per function. Now every query comes
from a fixed sequence:
1. Rebuild the history from the raw portal files.
2. Re-select each surrogate's kernel by cross-validation.
3. Classify each function as solved, noisy, improving or converged.
4. Choose a move type to match the class.
5. Write down a numerical prediction before submitting.

Step 5 turns every result into a test. Round 11 shows both sides:
- **Confirmed:** F4's x4 vertex predicted 0.7431 and returned 0.7440, a new best. F8's dim-4
  step came within 0.003 of its prediction.
- **Falsified:** F7's six-coordinate GP step predicted 2.447 and returned 2.202. F2's ridge
  vertex predicted 0.614 and returned 0.463.

So F7 goes back to a single coordinate and F2 to its best-measured point.

**Which variables drive the variation.** I computed this rather than guessed it. Each fitted
GP has one lengthscale per input, and its inverse-squared lengthscales give a relevance share
per dimension: in effect, a variance decomposition of the response.
- **F3:** x3 carries essentially all of it.
- **F2:** x1 carries 98% and x2 2%. F2 is a one-dimensional function in a
  two-dimensional box.
- **F7:** x5 (36%), x4 (23%) and x2 (19%) dominate, while x1 and x3 together carry 6%.
- **F8:** three of eight dimensions (x3, x1 and x7) carry 70% of the relevance.

Then I ran an actual PCA on my own portal queries for each function, and the comparison was
uncomfortable. For F8, the first principal component of where I have sampled loads mainly on
dims 6 and 5, two of the least relevant inputs (4% each). Most of my F8 query budget went into
directions that explain almost none of the output's variation. In PCA terms, my sampling
variance and the function's variance point in different directions.

**What to keep exploring and what to simplify.** PCA keeps the components that carry the
variance and drops the rest, and I now apply the same rule to coordinates:
- **Frozen:** F2's x2 is held at 0.9266. F3 moves only where the GP's mean says, and since
  only x3 matters there, that is effectively a one-dimensional search. F1 and F5 are frozen
  completely.
- **Still searched:** F7's x2 has a material single-coordinate gain left (+0.024 at 0.262, per
  the GP line scan), so that is this round's F7 move. F8's next move is dim 8. It is not highly relevant, but it is
  the surrogate's largest single-coordinate gain, and a −0.10 step is cheap.

Simplifying also means dropping a technique: joint multi-coordinate GP steps have now failed on
both F7 and F8. Off-axis predictions are the surrogate's least reliable part, so I now move
along one direction at a time.

**How this shapes the final round.** Round 13 is pure exploitation. There is no later round to
benefit from what an exploratory query would teach, so its value is zero. Each function will get
whichever point has the best evidence:
- **F1, F5:** their known optima.
- **F4:** the better of the incumbent and this round's x2 probe.
- **F7, F8:** this round's single-coordinate moves if they gain, otherwise the incumbent.
- **F2, F3, F6:** the noisy ones go to the point with the best *average* reading, not the
  single luckiest one.

This round's exploration is therefore three cheap one-coordinate probes (F4 x2, F7 x2, F8
dim 8).

**PCA insights applied to interpreting the results.** Two carry over directly:
- **Variance is where the information is.** A coordinate with a long lengthscale is a
  dimension I can stop spending queries on.
- **Redundancy deserves suspicion.** Repeated queries along one direction add almost nothing.
  My F5 history is dominated by x1 moves (PC1 loads 0.99 on x1), and once the corner was
  confirmed those moves were redundant.

The deeper lesson concerns the analysis, not the function. PCA of the inputs alone would have
described my *behaviour*, not the function's structure. Only by pairing the sampling directions
with the GP's relevance did the mismatch show, and that pairing is what decides which directions
the final query uses.

**Round-12 queries**
```
1 0.628541-0.628541
2 0.682637-0.926564
3 0.473437-0.533013-0.863879
4 0.415962-0.412500-0.363817-0.416634
5 0.999999-0.999999-0.999999-0.999999
6 0.450943-0.368189-0.636307-0.737863-0.002595
7 0.002673-0.261928-0.255873-0.290302-0.327805-0.603094
8 0.198086-0.192416-0.216581-0.138786-0.800003-0.350000-0.198013-0.793085
```
