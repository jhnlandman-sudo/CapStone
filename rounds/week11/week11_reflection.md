# Component 22.1 — reflection on round 11 (20 data points)

**How past patterns shaped this round.** Three round-10 patterns decided most queries:
- **F7: the same move again.** The GP posterior-mean step gained +0.133, far more than the
  one-coordinate x4 steps it replaced (+0.106, +0.075, +0.046, then a flat vertex). A coordinate
  exhausted alone can still pay jointly, so F7 repeats the joint move (predicted 2.447 ± 0.051).
- **F4: rebuilding the bracket.** The x4 probe at 0.425 set a new best of 0.7089, which means
  the peak lies *below* 0.428, not at it. Three points now bracket it, and the vertex at
  0.4166 predicts 0.743.
- **F8: reversing direction.** Moving dim 4 to zero cost 0.010, so this round moves it the
  other way, to 0.139.

The replicates mattered as much:
- **F3:** the record of −0.0258 came back as −0.0464, so it was a lucky draw.
- **F6:** −0.061 came back as −0.088, a real but narrow peak.
- **F2:** three readings at one point now span 0.598–0.726.

**Clusters and recurring regions.** Yes.
48 of the 80 portal queries lie within 0.1 of their function's best point. Some of that is my
exploitation, but in three functions the cluster is the real shape:
- **F2 is a ridge.** Every reading with x1 between 0.66 and 0.70 returns 0.54–0.73, whatever
  x2 is, from 0.12 up to 1.0. At x1 = 0.58 or 0.74 it collapses to 0.02–0.36. So this round
  fits a quadratic across the ridge and queries its vertex, x1 = 0.691.
- **F1 and F5 are single points.** One is a Gaussian centre, the other the corner (1,1,1,1).
  Both are confirmed by identical repeat readings.
- **F6 is a tight basin.** Points just 0.03–0.05 from the best one return −0.26 to −0.29.
  That is why this round moves only x2, to the vertex of the path that produced the peak.

F3 has no cluster I would claim: its nearby readings differ by less than its noise.

**What has proven less effective, and the adjustment.**
1. **Quadratic fits to noisy readings.** They failed three times on F2 and F3. I now average
   repeats before fitting, and fit along the direction the function actually varies (x1 on
   F2), not the one I happened to step in.
2. **Multi-coordinate moves on F8.** Two of them missed, so F8 now changes one coordinate at a
   time.
3. **Treating the GP's noise estimate as truth.** It puts F6's noise at 0.10, yet two readings
   at the same point differ by only 0.027. The GP is blurring a narrow peak into noise, so on
   F6 I steer by the local quadratic and use the GP only for ranking.

**The parallel with clustering.** It is close. DBSCAN calls a region a cluster only when
enough points sit densely inside it, and it labels isolated points as noise. That is now my test
for a "good region":
- **Repeated readings that agree mark a real cluster.** F5 and F1 are the clearest cases.
- **A single high reading surrounded by poorer neighbours is an outlier.** F3's −0.0258 was
  exactly that.

Choosing k in k-means corresponds to deciding how many separate basins a function has. With
sampling this concentrated, I can only say "at least one". Averaging replicates is computing a centroid rather than trusting one member.

**What plots would show, and how they guide the next round.**
- **Best value so far against round:** steps for F1 and F5 that went flat early; a steady,
  slowing climb for F8; a staircase for F7 that is still rising (1.31 → 1.80 → 2.05 → 2.17 →
  2.30). Remaining rounds go to F7 first, then F4.
- **F2's readings against x1:** a narrow vertical band near 0.68 and a sharp drop either side.
  The same readings against x2 would show nothing, which is the evidence that x2 can be
  ignored.
- **Replicate points:** vertical spreads at one input: about 0.13 on F2, 0.02–0.03 on F3
  and F6, zero on F1 and F5. That spread is the bar any future improvement must clear before I
  believe it.

So the next iteration spends where the staircase still rises, and treats any single noisy
reading as provisional until repeated.

**Round-11 queries**
```
1 0.628541-0.628541
2 0.691064-0.926564
3 0.444791-0.562256-0.859194
4 0.415962-0.417500-0.363817-0.416634
5 0.999999-0.999999-0.999999-0.999999
6 0.450943-0.341626-0.636307-0.737863-0.002595
7 0.000391-0.222653-0.297327-0.290152-0.376036-0.554072
8 0.198086-0.192416-0.216581-0.138786-0.800003-0.350000-0.198013-0.893085
```
