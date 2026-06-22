# ResidualClipAdamW-M on heavy-tailed quadratics

## Question

Can `ResidualClipAdamW-M` match clipped AdamW while being less sensitive to the
absolute clipping threshold, and is it stable enough to justify an ALBERT-base-v2
RTE sweep? The initial experiments found a moment mismatch. The implementation
now uses the same residual-centered pseudo-gradient for both Adam moments; this
report keeps the pre-change results for comparison and adds a matched post-change
evaluation.

## Protocol

The experiment uses the production `AdaptiveAdamW` implementation, rather than a
separate optimizer approximation. The objective is

```text
f(x) = 0.5 * x^T H x
```

with 32 logarithmically spaced eigenvalues from 1 to 100 and `||x_0|| = 5`.
The stochastic gradient is `H x + noise`. Noise has a uniform random direction
and a symmetric Pareto radius. Tail indices 1.2 and 1.5 have infinite variance;
2.5 is the finite-variance control. Every method sees the same initial point and
noise path for each seed.

The main comparison uses 2,000 steps, five seeds, learning rate 0.01, Adam betas
`(0.9, 0.999)`, global clipping, and thresholds
`0.03, 0.1, 0.3, 1, 3, 10, 30`. In addition to trajectories, the runner logs:

- tail, final, best, and log-AUC objective;
- objective spikes, divergence, and descent-alignment rate;
- median, p99, and maximum normalized update norm;
- response to the largest Pareto outlier and noise/update rank correlation;
- first- and second-moment norms, pseudo-gradient and center norms;
- gradient and residual clipping scales and activation rates.

The complete ignored run artifacts are under `outputs/adaptive_quadratics/`.
The experiment entrypoint and compact default are:

```bash
python -m experiments.quadratics.adaptive
```

## Pre-change mechanism diagnostic

For a constant scalar gradient `g = 10`, parameters are held fixed with learning
rate zero. At step 1,000:

| method | C | m_hat | sqrt(v_hat) | normalized update |
| --- | ---: | ---: | ---: | ---: |
| clipped AdamW | 0.1 | 0.1 | 0.1 | 1.0 |
| clipped AdamW | 1.0 | 1.0 | 1.0 | 1.0 |
| ResidualClipAdamW-M before coupling | 0.1 | 9.998 | 0.1 | 99.98 |
| ResidualClipAdamW-M before coupling | 1.0 | 10.0 | 1.0 | 10.0 |

This is structural. M's residual-clipped first moment can recover the full mean
gradient, while its second moment continues to receive the raw gradient clipped
at `C`. The resulting Adam update scales approximately as `|g| / C` once the
center has tracked a persistent gradient. Clipped AdamW sends the same clipped
signal to both moments, so the scale cancels.

## Pre-change main results

The table reports median tail objective across five seeds and its spread over
thresholds. Lower is better.

| Pareto alpha | method | best | worst | log10 spread | thresholds within 2x best |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1.2 | clipped AdamW | 0.0254 | 0.0538 | 0.326 | 6 / 7 |
| 1.2 | ResidualClipAdamW-M | 0.0347 | 736.2 | 4.326 | 3 / 7 |
| 1.5 | clipped AdamW | 0.0226 | 0.0333 | 0.169 | 7 / 7 |
| 1.5 | ResidualClipAdamW-M | 0.0292 | 659.0 | 4.354 | 3 / 7 |
| 2.5 | clipped AdamW | 0.0163 | 0.0189 | 0.065 | 7 / 7 |
| 2.5 | ResidualClipAdamW-M | 0.0171 | 671.1 | 4.594 | 3 / 7 |

M matches clipped AdamW around `C = 3, 10, 30`, but fails badly for `C <= 1`.
Its median update norm varies by about 20x over the threshold grid, compared with
about 1.3x for clipped AdamW. At small thresholds its descent-alignment rate is
near 0.5, so the large updates are also poorly directed.

No run crossed the numerical divergence cutoff at learning rate 0.01. This does
not make M stable in the useful optimization sense: at `C = 0.03` its objective
grew to roughly four times its initial value and then remained hundreds of loss
units above the clipped baseline.

## Coupled-moment change

`ResidualClipAdamW-M` now constructs one pseudo-gradient and uses it for both
moments:

```text
c_t = m_hat_(t-1)
p_t = c_t + clip(g_t - c_t, C)
m_t = beta1 * m_(t-1) + (1 - beta1) * p_t
v_t = beta2 * v_(t-1) + (1 - beta2) * p_t^2
```

Previously, `v_t` received `clip(g_t, C)^2`. On the constant-gradient diagnostic
with `g = 10` and `C = 0.1`, the normalized update at step 1,000 falls from about
100 to 1.47. This removes the `|g| / C` amplification mechanism.

The matched post-change main sweep gives:

| Pareto alpha | method | best | worst | log10 spread | thresholds within 2x best |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1.2 | clipped AdamW | 0.0254 | 0.0538 | 0.326 | 6 / 7 |
| 1.2 | coupled ResidualClipAdamW-M | 0.0278 | 55.90 | 3.303 | 3 / 7 |
| 1.5 | clipped AdamW | 0.0226 | 0.0333 | 0.169 | 7 / 7 |
| 1.5 | coupled ResidualClipAdamW-M | 0.0237 | 51.04 | 3.333 | 4 / 7 |
| 2.5 | clipped AdamW | 0.0163 | 0.0189 | 0.065 | 7 / 7 |
| 2.5 | coupled ResidualClipAdamW-M | 0.0129 | 48.08 | 3.570 | 3 / 7 |

This is a large improvement over the pre-change worst objectives of 659 to 736.
Median update norms at the smallest thresholds fall from about 19 to about 5.
For `C >= 3`, coupled M matches or slightly beats clipped AdamW. For `C <= 0.3`,
it remains poor because the residual center can move only by an amount proportional
to `C` on each step, so it tracks a changing gradient too slowly.

## Pre-change heavy-noise stress test

With Pareto alpha 1.2 and noise scale increased from 1 to 10:

| method | best tail objective | worst tail objective | log10 spread |
| --- | ---: | ---: | ---: |
| clipped AdamW | 0.281 | 0.347 | 0.091 |
| ResidualClipAdamW-M | 0.395 | 726.1 | 3.264 |
| uncut AdamW | 1.776 | 1.776 | n/a |

M and clipped AdamW both suppress the immediate response to the largest outlier.
Uncut AdamW's largest-outlier update is about 17x its median update, compared with
about 1x for both clipped methods. M therefore has local outlier resistance, but
its persistent update-scale mismatch dominates overall performance.

After moment coupling, the same stress test improves M's tail objective from
726.1 to 61.1 at `C = 0.1` and from 189.1 to 5.64 at `C = 1`. Its maximum update
norm is at most 6.70 across the tested thresholds, compared with values near 24
before coupling. No run diverged. At `C = 10, 30`, coupled M obtains 0.518 and
0.377, versus 0.281 and 0.347 for clipped AdamW.

## Pre-change learning-rate interaction

Learning rates `0.001, 0.003, 0.01, 0.03` were crossed with thresholds
`0.1, 1, 3, 10` at Pareto alpha 1.5. Even after selecting the best learning rate
separately at every threshold, M's best tail objective ranges from 0.0034 to
278.6, an 81,900x ratio. Clipped AdamW ranges from 0.0048 to 0.0109, a 2.3x ratio.
The threshold problem cannot be repaired by one shared learning-rate change.

## Candidate changes

The safest minimal change is moment coupling: construct the residual-centered
pseudo-gradient once and feed that same tensor to both `m` and `v`. This is now
implemented in `ResidualClipAdamW-M`. It removes M's unbounded constant-gradient
ratio and improves the main worst-case spread from about 4.5 to about 3.4 decades.
It still performs poorly for `C <= 0.3`, because a residual center driven by
bounded increments adapts at a rate proportional to `C`.

The existing raw-gradient-v, variable-alpha, and metric variants do not solve the
problem in the screening run. Their threshold spreads are roughly 3.8, 5.4, and
5.8 decades, respectively. In particular, variable-alpha bias correction can
produce normalized update norms in the hundreds when its accumulated bias mass
is small.

A next version should satisfy these invariants before an RTE sweep:

1. Use one robust pseudo-gradient for both Adam moments. The coupled M version now
   satisfies this invariant.
2. Replace the absolute residual threshold with a robust online residual-scale
   estimate and a dimensionless multiplier or target clipping rate. This removes
   the `1 / C` update amplification and the `C`-proportional center tracking speed
   from the user-facing tuning parameter.
3. Warm up the scale estimate with clipped AdamW and log moment consistency,
   `||m_hat / sqrt(v_hat)||`, clipping rate, and center tracking lag.
4. Require the infinite-threshold limit to match AdamW and require a constant
   gradient test to keep normalized updates bounded independently of threshold.

## Recommendation

The coupled `ResidualClipAdamW-M` is materially safer and is reasonable as an
exploratory ALBERT/RTE comparison, particularly near thresholds where clipping is
not continuously active. It should not yet replace clipped AdamW as the reliable
baseline, and it does not validate the desired threshold-insensitivity claim.
The next necessary change is a robust, self-scaled residual threshold that removes
the remaining `C`-proportional center tracking speed.

## Learning-rate and momentum sweep

The coupled implementation was also evaluated in a joint hyperparameter sweep:

- methods: AdamW, AdamWClip, and `ResidualClipAdamW-M`;
- thresholds: `0.03, 0.1, 0.3, 1, 3, 10, 30`;
- learning rates: `0.001, 0.003, 0.01, 0.03`;
- first-moment coefficients: `0.5, 0.7, 0.9, 0.95, 0.99`;
- second-moment coefficient: `0.999`;
- five matched seeds, 2,000 steps, Pareto alpha 1.5, and noise scale 1.

This is 1,500 optimizer runs. For each seed, performance is the median objective
over the final 20 percent of steps. Hyperparameters are selected by the mean of
that value across seeds. Plot bands show the seed interquartile range.

| C | best AdamWClip loss | lr | beta1 | best ResidualClipAdamW-M loss | lr | beta1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 0.00738 | 0.003 | 0.70 | 10.813 | 0.003 | 0.50 |
| 0.1 | 0.00738 | 0.003 | 0.70 | 0.255 | 0.003 | 0.50 |
| 0.3 | 0.00738 | 0.003 | 0.70 | 0.0176 | 0.003 | 0.50 |
| 1 | 0.00737 | 0.003 | 0.70 | 0.00808 | 0.003 | 0.50 |
| 3 | 0.00825 | 0.003 | 0.70 | 0.00644 | 0.003 | 0.90 |
| 10 | 0.0101 | 0.003 | 0.50 | 0.00860 | 0.003 | 0.95 |
| 30 | 0.0106 | 0.003 | 0.50 | 0.0111 | 0.003 | 0.99 |

The best AdamW configuration is `lr=0.003, beta1=0.99`, with loss 0.0214.
The globally best AdamWClip configuration is `C=1, lr=0.003, beta1=0.7`, with
loss 0.00737. The globally best residual configuration is
`C=3, lr=0.003, beta1=0.9`, with loss 0.00644.

After learning-rate and momentum tuning, AdamWClip changes by only 1.43x over the
threshold interval. ResidualClipAdamW-M changes by roughly 1,680x because of
`C=0.03`; over the narrower interval `C=1` to `30`, it changes by 1.72x and is
competitive with AdamWClip. Thus moment coupling makes the method useful over a
substantial threshold range, but the experiment still does not support the claim
of lower threshold dependence over the full interval.
