# Component 24.1 — final-round reflection through a reinforcement-learning lens

**How my exploration–exploitation balance evolved.** I treated the capstone as a multi-armed
bandit in which each round let me pull one arm per function. My exploration weight fell as the
data grew:
- **Rounds 1–3 (very little data):** GP-UCB with a moderate exploration weight (beta 0.5–0.8).
  This rewards points where the model is uncertain, and it found the regions that mattered.
- **Rounds 4–10 (structure visible):** shrinking trust regions, single-coordinate moves and
  three-point quadratic vertices. By round 10, 48 of my 80 portal queries lay within 0.1 of the
  incumbent.
- **Round 13 (final):** pure exploitation. With no later round, an exploratory query has no
  value, so every function goes to its best-evidenced point.

I kept taking risks the data justified: the F5 corner jump gained 84%. Joint six-coordinate
steps on F7 failed (predicted 2.447, returned 2.202), so I reverted to one-coordinate moves.
Round 12 tested that: F7's x2 probe returned 2.390 (new best) and F8's dim-8 probe 9.930 (new
best), while F4's x2 probe fell to 0.739, so only the first two are kept.

**How feedback shaped the process.** Each round returned one scalar reward per function: sparse,
evaluative feedback that says how good a choice was, not what the right choice was. I wrote a numerical prediction before every query. In effect,
the difference between result and prediction is a temporal-difference error, and I used it as
Q-learning does: Q ← Q + α(r − Q).
- **Small errors** (F4 predicted 0.7431, got 0.7440) confirmed the value estimate, so I kept
  the policy.
- **Large errors** (F2's ridge vertex predicted 0.614, got 0.463) forced a policy change, not
  just a new point.

The learning rate also depended on noise. F2's fourth replicate returned 0.460, so its four
readings average 0.608: the 0.726 was luck. Noisy functions are therefore ranked by mean
minus its standard error, as a bandit agent averages rewards per arm.

**Feedback and efficiency.** Some feedback was effectively delayed. A replicate earns nothing
in the round it is spent, but its reward arrives later: once F2 was shown to be noisy, it
stopped me fitting curves to noise for the rest of the challenge. That is credit assignment
over time in an MDP: an action's value includes what it makes possible later, so information
gathering paid early and is worthless in round 13. Efficiency also came from GP lengthscales: three of F8's eight inputs carry about 70% of the
relevance, so I stopped spending queries on the rest.

**Iterative improvement and the AlphaGo Zero analogy.** AlphaGo Zero improved by playing against
its own current network, with tree search turning each game into better training targets. My loop had the same shape on a small scale:
- **Plan with a model.** The GP surrogate, re-tuned by cross-validation each round, acted as a
  learned simulator: thousands of candidates were evaluated inside it before one real query.
- **Learn from trial and error.** Where the model was unreliable, especially off-axis in high
  dimensions, I fell back on model-free probes: one coordinate, one query, keep it only if the
  reward improves.
- **Test against your own predictions.** Measuring each outcome against my last prediction is
  the closest analogue to self-play. Falsified predictions changed the method; confirmed ones
  narrowed the search.

The lesson matches AlphaGo Zero's: a model makes planning cheap; real feedback keeps it honest.

**Applying RL principles to real-world optimisation.** For calibrating actuarial models, where
one run takes minutes to hours, four principles carry over:
- **Schedule exploration explicitly.** Let the exploration budget decay as remaining
   evaluations fall.
- **Treat value estimates as uncertain.** Replicate noisy evaluations and average them.
- **Use a model where it is trustworthy and trial and error where it is not.** Plan with a
   surrogate, then confirm with real runs, as Dyna-style methods do.
- **Log predictions.** A recorded prediction turns every result into a reward-prediction
   error that decides whether to adjust parameters or strategy.

**Final queries (round 13)**
```
1 0.628541-0.628541
2 0.682637-0.926564
3 0.444791-0.562256-0.859194
4 0.415962-0.417500-0.363817-0.416634
5 0.999999-0.999999-0.999999-0.999999
6 0.450943-0.365418-0.636307-0.737863-0.002595
7 0.002673-0.261928-0.255873-0.290302-0.327805-0.603094
8 0.198086-0.192416-0.216581-0.138786-0.800003-0.350000-0.198013-0.793085
```
