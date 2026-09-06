# Direction vs. magnitude: a two-mode toy on $\mathcal{S}^2$

This is the controlled experiment behind the paper claim:

> Under high CFG, geolocation sampling is essentially **basin selection** on a
> compact manifold: **direction chooses the basin; magnitude merely chooses the
> speed within it.** We verify this mechanism in a controlled toy experiment on a
> two-mode velocity field on $\mathcal{S}^2$ (Figure~\ref{fig:toy}), where
> angle-rotated perturbations of the field displace endpoints far more than
> norm-scaled perturbations of equal $L^2$ size.

It explains *why* the trajectory-deviation attack targets **cosine deviation**
from the predicted velocity rather than **$L^2$ norm deviation**.

## The mechanism

PLONK's base models are Riemannian flow matching (RFM) on the sphere: the network
predicts a **velocity** $v_\theta(x,\gamma)\in T_x\mathcal{S}^2$ and sampling
integrates $\dot x = v_\theta(x,\gamma)$ with an Euler step + reprojection
(`riemannian_flow_sampler`). For a fixed conditioning the velocity field is
**multimodal** (several plausible locations); each mode is a basin of attraction,
and the sampled endpoint is whichever basin the trajectory falls into.

Two ways to perturb the velocity, at equal $L^2$ size:

- **Magnitude (norm) deviation** $v\mapsto(1+c)\,v$. For a deterministic field a
  positive rescaling only **reparameterizes time** — it leaves the integral curve
  (the path) unchanged and merely changes how fast you move along it. It cannot
  move the endpoint off its basin; at most it over/undershoots the mode by the
  residual, un-traversed arc. This is what an $L^2$ objective mostly buys.
- **Direction (cosine) deviation** $v\mapsto R_\theta v$ (rotation in the tangent
  plane). This changes the **path itself**. Near a separatrix (the set
  equidistant between two modes) an arbitrarily small rotation flips the
  trajectory into the *other* basin, displacing the endpoint by ~the full
  inter-mode distance.

So the same $L^2$ perturbation budget buys a structural, basin-changing move when
spent on **angle** and only a within-basin speed change when spent on **norm** —
which is exactly why the attack maximizes cosine deviation from the predicted
velocity.

## What the experiment does

1. Builds a **two-mode** target on $\mathcal{S}^2$ (two tight blobs) and fits a
   **tiny RFM** (a 3-layer MLP velocity net) to it, **reusing the real PLONK
   geometry and objective**:
   - `plonk.utils.manifolds.Sphere` — manifold (logmap/expmap/projx),
   - `plonk.models.losses.RiemannianFlowMatchingLoss` — the exact training loss,
   - `plonk.models.schedulers.SigmoidScheduler` — the time schedule,
   - `plonk.models.samplers.riemannian_flow_sampler` — the sampler our integrator
     mirrors.
2. Sweeps a **matched $L^2$ budget** $s$ and, for starts in the basin-selection
   regime (near the separatrix), measures endpoint displacement and basin-flip
   rate under **angle** vs **norm** perturbations. Matching is exact per step:
   rotating by $\theta$ changes $v$ by $\lVert v\rVert\,2\sin(\theta/2)$ and
   rescaling by $1+c$ by $\lVert v\rVert\,|c|$, so $|c| = 2\sin(\theta/2) = s$
   makes the two perturbations identical in $L^2$ size (see
   `theta_for_strength`).
3. Exports **ensembles of trajectories** from many starts that nominally commit to
   basin A, integrated under nominal / direction-deviated / magnitude-deviated
   perturbations, so the figure shows *where endpoints land* (basin A vs B), plus
   one representative trajectory with its velocity arrows. It also exports **true-
   flow streamlines**: generative trajectories $\dot x=v_\theta(x,\gamma)$ of the
   *unperturbed* field, seeded on an even grid over the visible hemisphere, shown
   faint in the TikZ sphere so the two-basin structure is visible under the
   perturbed ensembles. (They can cross in projection because the field is
   time-dependent in $\gamma$ — that is expected, not a bug.)

## How the perturbations are crafted (exactly)

Sampling integrates $\dot x = v_\theta(x,\gamma)$ with the manifold Euler step
$x_{t+1} = \mathrm{proj}_{\mathcal S^2}\!\big(x_t + \Delta\gamma_t\, \tilde v_t\big)$,
$\Delta\gamma_t=\gamma_{t+1}-\gamma_t$. At every step we take the model velocity
$v = v_\theta(x_t,\gamma_t)\in T_{x_t}\mathcal S^2$ and replace it with a perturbed
velocity $\tilde v$ **before** the step — a *closed-loop* perturbation, re-evaluated
at the actual (drifted) state $x_t$, exactly as the trajectory-deviation attack
perturbs the predicted velocity at each sampling step. Two families
(`toy_rfm.py`):

- **Direction (cosine) deviation** — rotate $v$ inside the tangent plane by angle
  $\theta$ about the outward normal $x$ (Rodrigues):
  $\tilde v = \cos\theta\,v + \sin\theta\,(x\times v)$ (`rotate_in_tangent`).
  $x\times v$ is $v$ turned a quarter-turn in $T_x\mathcal S^2$ with the same
  length, so this **preserves $\lVert v\rVert$ and tangency** and changes only the
  direction. Per step: $\lVert\tilde v-v\rVert=\lVert v\rVert\,2\sin(\theta/2)$.
- **Magnitude ($L^2$/norm) deviation** — rescale, keep the direction:
  $\tilde v = (1+c)\,v$ (`norm_perturbation`). Per step:
  $\lVert\tilde v-v\rVert=\lVert v\rVert\,|c|$.

**Matched $L^2$ budget.** One relative per-step size $s$ drives both: set $|c|=s$
and $\theta = 2\arcsin(s/2)$ (so $2\sin(\theta/2)=s$, `theta_for_strength`). Then
the per-step perturbation vectors have **identical $L^2$ size** $s\lVert v\rVert$
for both families, hence identical total-trajectory $L^2$. Any endpoint difference
is due to *where* the same-sized budget is spent (angle vs norm), not its size.

**Worst sign.** The attacker picks the sign, so displacement/flip take the max over
$\pm\theta$ and over $(1\pm s)$ (the magnitude factor floored at $0.3$ so a
degenerate full stop $v\to 0$ is not counted as a magnitude perturbation).

**Two regimes.** The aggregate sweep applies a **constant** per-step rotation (the
rigorous worst-case sensitivity). The figure ensembles/representative use an
**early-window** rotation (rotate only during the first 40% of steps, then follow
the clean field), so the flipped trajectory lands *on* mode B instead of
spiralling — the "basin chosen early near the separatrix, then carried in" picture.
Magnitude in the figure is the worst-sign rescale.

## How to run

Use the project conda env (has `plonk` + `geoopt`):

```bash
cd adversarial_demo/toy_example
PY=/Data/mathias.ollu/conda/plonk/bin/python

# 1. train the tiny RFM + run the experiment  (~5-7 min on CPU; caches out/toy_rfm.pt)
$PY run_toy_experiment.py

# 2. render the figure (matplotlib -> out/toy_figure.pdf/.png)
$PY plot_toy.py

# 3. (optional) emit a standalone TikZ sphere for the paper
$PY export_tikz.py
cd out && pdflatex toy_sphere.tex
```

For the paper, use the matplotlib `toy_figure.pdf` (three globes + curves)
directly, or drop in the TikZ sphere: `\includegraphics{toy_sphere.pdf}` the
compiled standalone, or copy the `tikzpicture` body out of `out/toy_sphere.tex`
into a `figure` environment (the preamble needs `\usetikzlibrary{arrows.meta}`
and `amsmath`).

Everything runs on CPU. Re-running reuses the cached model; pass `--retrain` to
refit. Useful flags: `--steps` (training steps), `--num-steps` (sampling Euler
steps), `--n-starts` (sweep size), `--s-hero` (matched budget / ensemble budget).

## Files

| file | role |
|------|------|
| `toy_rfm.py` | geometry helpers, the tiny RFM (`ToyRFM`), training via the real `RiemannianFlowMatchingLoss`, and the perturbable integrator (`integrate`, `angle_perturbation`, `norm_perturbation`, `theta_for_strength`). |
| `run_toy_experiment.py` | trains, runs the matched-$L^2$ sweep, builds the trajectory ensembles, writes `out/toy_results.json` + `out/toy_geometry.npz`. |
| `plot_toy.py` | matplotlib figure: three shaded globes (nominal / direction- / magnitude-deviated ensembles, endpoints coloured by basin) above the displacement + basin-flip curves. |
| `export_tikz.py` | self-contained `out/toy_sphere.tex`: one clean light ball with both ensembles (direction→B in blue, magnitude→A in red), velocity arrows, and the faint true-flow streamlines. Orthographic projection done in Python; plain TikZ, no raster. |

## Outputs (`out/`)

- `toy_results.json` — matched-$L^2$ sweep curves (`angle_*` vs `norm_*`
  displacement, realized $L^2$, basin-flip fractions) + headline numbers.
- `toy_geometry.npz` — modes, separatrix, background streamlines, the three
  trajectories, and the velocity arrows at the marked point.
- `toy_figure.pdf/.png` — the matplotlib figure (three globes + curves).
- `toy_sphere.tex/.pdf` — the standalone TikZ sphere.

## Result (reproduced run)

Two modes 104° apart (inter-mode distance 1.82 rad), tiny RFM trained 8000 steps,
250-step sampler.

**Ensemble** (36 starts that all nominally commit to basin A, matched budget
$s=0.5$): nominal → **0%** in B; direction-deviated → **86%** migrate to B;
magnitude-deviated → **8%** (stays at A). This is the top row of the figure.

**Representative trajectory** (one separatrix start, matched budget $s=0.5$):

| perturbation | endpoint displacement | basin |
|--------------|----------------------:|-------|
| direction (rotate $v$) | **1.70 rad** (lands 0.10 rad from mode B) | **flips A→B** |
| magnitude (rescale $\|v\|$) | 0.06 rad | stays in A |

→ a **~27×** displacement gap at identical $L^2$ size.

**Aggregate** over 48 separatrix-regime starts, worst-case over the perturbation
sign, per matched budget $s$:

| $s$ | dir. disp | mag. disp | dir. basin-flip | mag. basin-flip |
|-----|----------:|----------:|----------------:|----------------:|
| 0.2 | 0.28 | 0.09 | 12% | **0%** |
| 0.4 | 0.53 | 0.25 | 23% | **0%** |
| 0.6 | 0.75 | 0.57 | 31% | **0%** |
| 0.8 | 0.90 | 0.85 | 35% | 4% |
| 1.0 | 1.10 | 0.85 | 42% | 4% |

The decisive quantity is the **basin flip** (a confident *wrong* location):
**magnitude perturbations essentially never change the basin** (0% up to $s=0.7$),
while direction perturbations change it with increasing frequency. Magnitude does
move the endpoint — but only *within* the basin (changing arrival speed/position),
and that displacement **saturates** (bounded by the residual arc to the mode),
whereas direction displacement keeps growing toward the full inter-mode distance.
This is exactly why the trajectory-deviation attack maximizes cosine deviation
from the predicted velocity rather than its $L^2$ norm.

(Numbers regenerate into `out/toy_results.json`; the exact values shift slightly
with seed/among runs but the picture is stable.)
