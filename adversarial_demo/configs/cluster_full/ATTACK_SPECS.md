# cluster_full run — attack specifications (as implemented) vs. the paper

Extracted from the code paths that `configs/cluster_full/config.yaml` dispatches to
(`attacks/attacks.py::run_attack` — each key is one fixed mechanism, `attacks/trajectory_deviation.py`,
`attacks/encoder_attacks.py`, `attacks/diffusion_attack_salman.py`, `attacks/attacks_core.py`).
Paper references are to *Geographic Flow Attacks* (ICLR 2026 submission): Eq. 1 (forward
process), Eq. 5 (end objective), Eqs. 7–12 and 15 (surrogates), Sec. 6.1 (protocol).

## Common to every trainable attack (Sec. 6.1)

| item | value | paper |
|---|---|---|
| perturbation | δ on the [0,1] pixel scale, applied to the model-input square (CenterCrop 1:1 → 336 px), protected image = clamp(c+δ) quantised to 8 bit, re-encoded exactly like a clean image | Eq. 5, budget b on ‖δ‖∞ |
| budgets | 1/255, 4/255, 8/255 (`eps_max` = 0.0039 / 0.0157 / 0.0314) | b ∈ {4/255, 8/255} (+1/255 new) |
| optimiser | sign-SGD (PGD): δ ← clip(δ − lr·sign(∇), ±b); lr = 1e-3 (= 0.255 grey level / step), 60 steps | "60 PGD steps, lr 1e-3" |
| init | δ₀ ~ U(−1e-4, 1e-4) | – |
| Monte-Carlo batch | 256 (x_t, ε, t) samples per PGD step, t ~ U[0,1], γ = SigmoidScheduler(−7, 3, τ=1)(t) (the model's own schedule) | "batch size 256" |
| restarts | 5 independent inits per image and budget; **kept = highest FSD**, i.e. mean haversine between the attacked prediction and the TRUE location (`restart_selection_metric: final_step_displacement_true`; each restart evaluated with 250 steps, w=10, 128 samples, shared noise seed 42+restart) | 5 restarts, "select the one achieving the highest FSD" ✔ |
| reported metric | FSD = mean over 128 samples of haversine(attacked prediction, true GPS); 250 steps, w = 10, shared noise seed 42 | Sec. 6.1 |
| model | RFM on S² (`nicolas-dufour/PLONK_YFCC` with DINOv2 ViT-L/14-reg, `PLONK_OSV_5M` with StreetCLIP ViT-L) | Sec. 3 |

## Per attack

### `dtd` — GeoFA-TD (Eq. 15)
- Objective (minimised): E[ ⟨ψ(x_t|c+δ), ψ(x_t|c)⟩ / (‖ψ(x_t|c+δ)‖ ‖ψ(x_t|c)‖) ] — signed cosine similarity between perturbed and clean velocity predictions.
- Reference: the clean model's own prediction ψ(x_t|c) (conditional branch only, no CFG).
- **x_t: legacy Gaussian interpolant x_t = √γ·x₀ + √(1−γ)·n, n ~ N(0, I₃), with x₀ = the image's TRUE location** (unit vector). x_t is *off the sphere*; the cosine is the plain Euclidean one in R³ (no tangent projection).
- ⇒ **D2 (kept by decision)**: the paper writes Eq. 15 with x_t from Eq. 1 (geodesic exp_{x₀}(κ(t) log_{x₀} ε), ε ~ U(S²)) and ⟨·,·⟩_{x_t}; that geodesic/tangent version (option `forward_process: model`) scored lower in the 100-image study (11 893 vs 13 788 km at 4/255). The run keeps the Gaussian-interpolant version; the paper's forward-process description (Eq. 1 as used by the attack, and the ⟨·,·⟩_{x_t} notation in Eqs. 12/15) must be updated accordingly — GeoFA-L² uses the same interpolant so Sec. 5.2's "metric-only" comparison holds.

### `diffusion_l2` — GeoFA-L² (Eq. 12)
- Objective (maximised): E ‖ ψ(x_t|c+δ) − ψ(x_t|c) ‖² (Euclidean, mean over batch and the 3 coordinates).
- **Same forward process as the TD pin**: x_t = √γ·x₀ + √(1−γ)·n, n ~ N(0, I₃), x₀ = TRUE location; raw (unprojected) velocity predictions. TD and L² therefore differ by the metric only (cosine vs L²), as Sec. 5.2 argues.
- Paper: Eq. 12 objective ✔; the forward process is the one of D2 (`forward_process: model` gives the geodesic/tangent version as an ablation).

### `targeted_l2` — GeoFA-DPA / L²-targeted (Eq. 11)
- Target x'₀ = antipode of the TRUE location, −cart(x₀).
- x'_t = geodesic(ε ~ U(S²) → x'₀)(γ), v(x'_t) = ∂_γ geodesic (the training target of the model).
- Objective (minimised): E ⟨ψ(x'_t|c+δ) − v(x'_t), ·⟩_{x'_t} / 3 — the RFM training loss towards the antipode (DP-Attacker Eq. 6, targeted branch). ✔ Eq. 11 (the paper's "x'_t = x'₀ + ε" is the loose Euclidean form).
- Label in figures/tables: "GeoFA-DPA" (the paper's name for this instance).

### `sampling` — GeoFA-Sampling (Eq. 8)
- Objective (maximised): E_{x_N} haversine( x̂₀^{c+δ}(x_N), x₀_true ), backpropagated through the **full 16-step** projected-Euler RFM sampler; a fresh batch of 256 x_N ~ N(0, I₃) per PGD step.
- The attacked 16-step sampler runs **with guidance w = 10** (`sampling_cfg` default 10 = the evaluation guidance; the sampler doubles the batch with a zero embedding for the unconditional branch, gradients flow through the conditional one). ✔ Eq. 8 / App. C.2 ("16 timesteps"). (The 100-image study ran it at w = 0, option `sampling_cfg: 0`.)

### `encoder` — GeoFA-Encoder (Eq. 7)
- Objective (maximised): ‖E(c+δ) − E(c)‖²₂, E = the CLS embedding actually fed to the velocity network (DINOv2 / StreetCLIP), encoder in eval mode.
- Same sign-SGD loop as the others (option `update_rule: adam` = the original Adam / full-budget-init version). Untargeted. ✔ Eq. 7.

### `training_loss` — GeoFA-AdvDM (Eq. 9)
- x_t = geodesic(ε ~ U(S²) → x₀ = TRUE location)(γ), v(x_t) = ∂_γ geodesic — exactly `RiemannianFlowMatchingLoss` of PLONK.
- Objective (maximised): E ⟨ψ(x_t|c+δ) − v(x_t), ·⟩_{x_t} / 3 (uniform weighting). ✔ Eq. 9 with Eq. 1.

### `unidef` — GeoFA-UniDef (Eq. 9 + FDJE, App. C.3)
- Same objective as AdvDM (training loss, true target v(x_t), geodesic x_t, uniform w(t)), gradient estimated by a **symmetric finite difference** in embedding space: z = E(c)/‖E(c)‖, h = 0.01, ĝ_emb = 2⟨(ψ(x_t, e+hz) − ψ(x_t, e−hz))/2h, ψ(x_t,e) − v⟩ z, back-propagated to δ through the encoder only. ✔ Eq. 20.
- x₀ = TRUE location, so it differs from GeoFA-AdvDM by the gradient estimator only. ✔ "same quantity, different estimator" (App. C.3).

### `dtd_blur` — GeoFA-TD-blur
- (1−w)·L_TD(c+δ) + w·L_TD(blur_σ(c+δ)), w = 0.5, σ ~ U[0, 4] per step, differentiable Gaussian blur (25×25 kernel, reflect padding) of the protected image; reference ψ(x_t|c) stays the *un-blurred* clean prediction; L_TD identical to `dtd` above (Gaussian x_t, true x₀, Euclidean cosine).
- Not described in the paper body (robustness ablation candidate).

### `geoshield` — GeoShield (external, Liu et al. 2025)
- Released `Geoshield/geoshield.py`, config `ensemble_3models`: FGSM-style sign steps, α = 1 grey level, ε ∈ {1, 4, 8} grey levels, 100 steps, `input_res` 336 (budget applied at the attacked model's resolution), losses = −(global + random-224-crop) CLIP-ensemble feature similarity to the clean image (CLIP ViT-B/16, ViT-B/32, LAION ViT-G/14, random resized crops scale [0.5, 0.9]) − geo-text loss.
- Clean copies: lossless PNG of the attacked pipeline's own preprocessed square (model-exact), so GeoShield's crop/resize is a no-op and the perturbation is measured on the evaluator's clean tensor; the protected PNG is scored exactly like the trainable attacks (250 steps, w = 10, 128 samples).
- ⇒ discrepancy **D5**: the released code ships a *placeholder* VLM description ("A scenic outdoor photograph.") and no detector boxes (`bbox_json_path` empty → the only "region" is the full image). Our GeoShield is therefore the CLIP-ensemble feature attack with a constant geo-text term, not the full "CLIP ens. + VLM / det." pipeline of Table 1. This was already the case in the previous full runs.

## Discrepancies to resolve before / while writing

| id | what | options |
|---|---|---|
| D1 | restart selection | **resolved**: `restart_selection_metric: final_step_displacement_true` — the restart with the highest FSD to the true location is kept |
| D2 | GeoFA-TD and GeoFA-L² use the Gaussian interpolant √γ x₀ + √(1−γ) n (x₀ = true location) and Euclidean metrics; Eq. 1/12/15 describe the geodesic + tangent version | **decided: keep** — rewrite the attack-side forward process in the paper (Eq. 1 is the *model's* process; the attacks noise the true location with the Gaussian interpolant) |
| D3 | UniDef x₀ source | **resolved**: true x₀ |
| D4 | sampling attack guidance | **resolved**: trained through the sampler at w = 10 |
| D5 | GeoShield without VLM description / detector regions | known; state it in the GeoShield paragraph |
| – | 1/255 budget and the JPEG q=100 level are new w.r.t. the paper's grids | update Sec. 6.1 / Fig. 5 |
