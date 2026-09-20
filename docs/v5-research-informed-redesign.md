# IgnisAI — Research-Informed Redesign

*What the wildfire-spread ML literature actually knows as of mid-2026, where IgnisAI
diverges from it, and what I'd do differently if starting over.*

---

## 0. The headline

Your v4 gameplan's thesis is: **v3 fails on Palisades because the training data lacks
Santa Ana fires → get TS-SatFire, drop shortcut features, retrain.**

That thesis is *partly* right and worth executing. But the strongest recent evidence
says the biggest accuracy levers in this exact task are **not** data volume or
architecture size — they're training-protocol details that your plan currently leaves
on the table, and several of your planned changes point the wrong way.

The single most important paper for you is **Lahrichi et al., "Improved Wildfire Spread
Prediction with Time-Series Data and the WSTS+ Benchmark" (WACV 2026)**. It's the first
controlled, apples-to-apples comparison of the modeling strategies everyone has been
publishing, on the only public time-series benchmark with realistic
leave-one-year-out cross-validation. Their ablation is the most actionable table in the
field:

| Change removed from their best model | Test AP | Drop |
|---|---|---|
| **Their Res18-UNet (full)** | 0.455 | — |
| No pre-training | 0.456 | ~0% (on this subset) |
| **No focal loss** | 0.345 | **−24.2%** |
| **No AP as validation metric** | 0.321 | **−29.5%** |

The largest single gain in the whole study came from **selecting the model on the same
metric you report** — not from data, not from architecture. The second largest came from
swapping the loss to **focal**. Both are ~1-day changes for you.

---

## 1. What the field actually knows (2025–2026)

### 1.1 The benchmark and the numbers

**WSTS (WildfireSpreadTS)** — 607 fire events, western US, 2018–2021, 13,607 daily
multi-channel images, 23 channels at 375 m, evaluated with **12-fold
leave-one-year-out cross-validation**. **WSTS+** doubles it to 8 years (2016–2023),
1,005 events, 24,462 images.

Metric: **Average Precision (AP)**, not IoU/CSI/F1. Current published results:

| Model | Input days | Veg | Multi | All | Params |
|---|---|---|---|---|---|
| Res18-UNet (Gerard 2023, prior SOTA) | 1 | 0.328 | 0.341 | 0.341 | 14.3 M |
| **Res18-UNet (improved)** | 1 | 0.455 | **0.468** | 0.460 | 14.3 M |
| Res50-UNet | 1 | 0.457 | 0.459 | 0.451 | 32.5 M |
| SwinUNet | 1 | 0.432 | 0.437 | 0.424 | 27.2 M |
| SegFormer | 1 | 0.433 | 0.436 | 0.423 | 27.5 M |
| UTAE (Gerard 2023) | 5 | 0.372 | 0.350 | 0.321 | 1.1 M |
| UTAE (improved) | 5 | 0.452 | 0.459 | 0.433 | 1.1 M |
| **UTAE(Res18) — overall SOTA** | 5 | **0.478** | **0.477** | **0.475** | 14.6 M |

### 1.2 Six findings that should change your plan

1. **Metric alignment dominates.** Validating on AP instead of F1 was worth +29%.
   Your acceptance criteria and threshold sweeps are built around CSI/F1.

2. **Focal loss beats Dice/Jaccard/BCE** for this severe class imbalance (+24%).
   Your v4 keeps Dice + BCE + Tversky.

3. **Bigger and fancier architectures lose.** SwinUNet and SegFormer both
   *underperform* a well-tuned ResNet-18 UNet, and the overall SOTA (UTAE) has
   **1.1 M parameters** — 25× smaller than the transformers it beats. Your gameplan
   says "lift to SwinUNETR later if v4 plateaus." The evidence says that's the wrong
   escape hatch. This held even under random (easier) cross-validation, so it isn't
   just an overfitting artifact.

4. **More data barely helps.** Doubling the years of historical data (WSTS → WSTS+)
   "yields little improvement." The authors instead identify **cross-year domain
   shift** as the field's critical unsolved problem. This is the finding that most
   threatens your v4 thesis — see §2.1.

5. **More features can hurt.** Best AP was usually on the *Veg* or *Multi* subsets,
   not *All*. Their hypothesis: weather features at 27 km resolution versus fire masks
   at 375 m add dimensionality and noise faster than signal. **Your v4 plan adds
   channels** (13 → 18 dynamic).

6. **Time-series input does help** (T=5 > T=1), consistently. Your T=6 window is right.
   But the *fusion mechanism* matters: feature-level fusion with temporal attention
   (UTAE) beats data-level concatenation, and **relative positional encoding
   `[1..T]` beat day-of-year encoding (0.452 vs 0.419)**.

### 1.3 Where the frontier is moving

- **Physics-hybrid models.** Neural-parameterized probabilistic cellular automata keep
  a physically transparent three-state CA substrate (mass conservation, ignition
  dynamics) and use a multi-scale CNN to emit *spatially varying* parameters for
  spread probability, wind alignment, and slope. You get physical interpretability
  plus learned nonlinearity. PINNs for wildfire parameter learning are the other branch.
  Reviews consistently recommend hybrids over pure-ML or pure-physics.
- **Probabilistic forecasting.** Conditional flow matching for localized spread —
  output a *distribution* of plausible perimeters, not one heatmap. For an advisory
  product this is strictly better than a point estimate.
- **Simulation as a data source.** Google's **FireBench** (with the US Forest Service
  Fire Lab) generates high-fidelity simulated fire spread across 117 wind-speed/slope
  combinations. That's a pretraining/augmentation corpus that directly addresses your
  "Santa Ana events are <2% of data" problem without needing more real fires.
- **Industry = fusion, not better single models.** The **Technosylva + Pano AI**
  partnership (Feb 2026) integrates predictive fire-behavior modeling with real-time
  camera-based ignition detection, plus "Time to Arrival" estimates for critical
  assets. The commercial value is in the *operational loop* — confirmed ignition →
  prediction → asset impact — not in the raw next-day segmentation score.

---

## 2. Where IgnisAI diverges from the evidence

Ranked by expected value per unit of effort.

### 2.1 🔴 The v4 thesis needs a cheap test before you spend 48 GPU-hours

Your plan: TS-SatFire (+Santa Ana fires) → retrain → Palisades works. The WSTS+ result
says adding data volume yields little improvement, and that the real enemy is
**cross-year / cross-regime domain shift**. Palisades is exactly a domain-shift
failure, not a data-volume failure.

That doesn't invalidate your plan — adding Santa Ana events changes the *distribution*,
not just the *volume*, which is a different intervention. But it does mean:

- **Test the cheap fixes first.** Metric alignment + focal loss + pretrained encoder
  are ~2 days and, per the ablation, worth more than most data changes.
- **Measure domain shift explicitly** rather than assuming more data fixes it. Train
  on non-Santa-Ana, test on Santa Ana, and report the gap. That number is your actual
  research contribution.
- Your **Santa Ana 5× sampler up-weighting is the right instinct** — it's a
  distribution intervention, not a volume one. Keep it, and ablate it.

### 2.2 🔴 Metric and model selection

| | IgnisAI now | Evidence says |
|---|---|---|
| Val metric | CSI @ threshold sweep | **AP** (threshold-free), matched to test metric |
| Reported metric | CSI, IoU, Dice, Hausdorff | AP primary (comparable to literature) + your operational metrics secondary |
| Selection | best CSI checkpoint | best **AP** checkpoint |

Threshold sweeps are a symptom: AP integrates over all thresholds, so you stop
tuning an operating point during model selection and pick it once, at deployment,
from the calibrated curve. **Keep** Hausdorff — directional error is
operationally meaningful and the literature under-reports it. That's a genuine
strength of your eval design.

### 2.3 🔴 Loss function

You're planning Dice + BCE + Tversky(0.3/0.7) + wind-align. Evidence: **focal loss**
substantially beat Dice, Jaccard, and BCE (α = inverse positive-class frequency,
γ = 2). Recommendation: make focal the primary term, keep your wind-align
regularizer at 0.1 (it's a genuinely novel, physically-motivated addition and I'd
want to see its ablation), and drop or ablate the rest rather than stacking four
losses without evidence.

### 2.4 🟠 Architecture — stop planning to go bigger

`ARCH_VERSION v4` on ConvLSTM-UNet is fine for shipping. But the "lift to SwinUNETR if
it plateaus" line should be replaced with: **try UTAE(Res18) — temporal attention,
pretrained ResNet-18 encoder, relative positional encoding.** It's the current SOTA,
it's ~14.6 M params, and it beats every transformer variant tested. If you want an
architecture bet, that's the evidence-backed one.

Also: your ConvLSTM trains **from scratch**. The SOTA model is literally "same
architecture, pretrained ResNet-18 encoder swapped in." That's a cheap, high-prior
win.

### 2.5 🟠 Channel expansion cuts against the evidence

Going 13 → 18 dynamic channels assumes more signal. The benchmark found *All* features
usually underperformed *Veg*/*Multi*, attributed to resolution mismatch — and you have
the same mismatch (HRRR ~3 km, GridMET ~4 km, tiles at 500 m).

Your **removals are well-justified** (`impervious`/`population` as urban-edge shortcuts
is a real, diagnosed failure — good catch). The **additions** (VIIRS I4/I5) are
plausible because thermal is genuinely new information at native resolution. But run
the channel ablation as a *first-class experiment*, not as notebook cell 13. Your
per-channel ablation cell is the right tool — promote it to a gating experiment.

### 2.6 🟠 No leave-one-year-out cross-validation

The field standard is 12-fold LOYO-CV precisely because it simulates deployment
(predicting a year you've never seen). You have a single val split plus 5 preset
events. Your 5-preset OOD harness is *good* and more operationally realistic than most
papers — but it's ~5 data points, so it can't distinguish a real improvement from
noise. LOYO-CV on your merged corpus would give you error bars (the papers report
±0.08–0.09 AP std — differences smaller than that are noise).

### 2.7 🟢 What you're already doing right

- Delta targets, T=6 temporal window, wind-vector-aware rotation augmentation.
- Diagnosing `channel_dropout` zeroing wind — that's a genuine bug the literature
  wouldn't have caught for you.
- Isotonic calibration with a `model_sha256` binding.
- Hausdorff distance in eval (directional error).
- The OOD preset harness concept.
- Honest "what this does not fix" section in the gameplan.

---

## 3. If I were rewriting from scratch

### 3.1 Principle: benchmark-first, product-second

The core mistake in the current structure is that **research and serving are
entangled** — `services/tilesvc` shapes tile geometry, which shapes the dataset,
which constrains the model. That's why "re-tile to 64×64 @ 500 m on the EPSG:5070
grid" is a hard requirement in your ingestion spec: the product is dictating the
science.

Split them:

```
ignis-research/          # benchmarked, reproducible, comparable to papers
  benchmarks/            # WSTS, WSTS+, TS-SatFire adapters — native resolution
  models/                # UTAE, Res18-UNet, ConvLSTM baselines
  experiments/           # one config per run, versioned, with AP + LOYO-CV
ignis-serving/           # tilesvc, grid, calibration — consumes a model artifact
  contracts/             # channel schema, arch version — the ONLY coupling point
```

The contract between them is a **model card**: channel order, normalization ranges,
arch version, AP on each benchmark, calibration curve, and known failure modes. If
serving needs 500 m tiles, that's a resampling step in serving — not a constraint on
the science.

**Why this matters concretely:** right now you can't report a number that anyone
in the field can compare to. Adopting WSTS/WSTS+ as a secondary benchmark means
you can say "IgnisAI v5 gets AP 0.4x on WSTS, competitive with SOTA, *and* here's
our Santa Ana OOD result" — which is a far stronger claim than any CSI number on
5 presets.

### 3.2 The model I'd build

**Baseline (week 1):** UTAE(Res18) reproduction on WSTS. Get to ~0.47 AP. This is
your sanity check that the harness is correct — if you can't reproduce published
numbers, no downstream result is trustworthy.

**v5 (weeks 2–4):** UTAE(Res18) + your three fire-specific contributions:
1. **Wind-alignment regularizer** (yours — novel, physically motivated)
2. **Santa Ana / extreme-wind-regime sampler** (yours — a domain-shift intervention)
3. **VIIRS thermal channels** (I4/I5 at native resolution, not resampled down)

Train on TS-SatFire + WSTS+ merged. Evaluate with LOYO-CV *and* your 5-preset OOD
harness. Report AP primarily.

**v6 (research bet, pick one):**
- **Physics-hybrid:** neural-parameterized CA. Best fit for your problem because the
  physical substrate enforces the wind/slope behavior you're currently trying to
  coax out with a regularizer. Highest ceiling, highest effort.
- **Probabilistic:** conditional flow matching for perimeter distributions. Best fit
  for your *product* — an advisory tool should show a cone of uncertainty, not a
  single heatmap. This also honestly handles the "underestimates magnitude on
  day 2" limitation you already documented.
- **Simulation pretraining:** pretrain on FireBench synthetic spread, fine-tune on
  real fires. Directly attacks the "Santa Ana events are rare" problem.

My ranking for your situation: **probabilistic first** (product value + honest
uncertainty), physics-hybrid second (research value), simulation pretraining third
(highest variance).

### 3.3 The product I'd build

Technosylva + Pano AI is the signal here: the industry has concluded the value is in
**detection → prediction → asset impact**, integrated. A better next-day segmentation
score is not the differentiator.

For IgnisAI specifically, the highest-value product features aren't model
improvements at all:
- **Time-to-arrival at named assets** (structures, evacuation routes, substations)
  rather than a probability field. That's the decision-relevant output.
- **Calibrated uncertainty bands** on the perimeter.
- **Provenance on every prediction** — which weather source, how stale the FIRMS
  data is, what the model's OOD score is for this event. You already track
  `weatherQuality`; extend it into a full confidence signal.

That last one is a genuine differentiator and cheap: a model that says "this event is
unlike my training distribution, treat with caution" is more useful operationally
than one that's 0.02 AP better.

---

## 4. Concrete revisions to the v4 plan

Do these **before** spending 48 GPU-hours.

| # | Change | Effort | Why |
|---|---|---|---|
| 1 | Switch val metric + model selection to **AP** | 0.5 d | Largest single gain in the ablation (+29%) |
| 2 | Make **focal loss** primary (α=inv-freq, γ=2) | 0.5 d | +24% in the ablation |
| 3 | **Pretrained ResNet-18 encoder** | 1 d | The only difference between UTAE and SOTA UTAE(Res18) |
| 4 | **Relative positional encoding** `[1..T]` | 0.5 d | 0.419 → 0.452 AP |
| 5 | Report **AP on WSTS** alongside your CSI presets | 1 d | Makes you comparable to the literature |
| 6 | Run **channel ablation as a gate**, not a notebook cell | 1 d | Evidence says added features may hurt |
| 7 | **LOYO-CV** for error bars | 1 d | ±0.08 std means your 5-preset deltas may be noise |
| 8 | Drop "lift to SwinUNETR" → **"try UTAE(Res18)"** | — | Transformers lose to well-tuned ResNet UNets here |

Steps 1–4 are ~2.5 days and, on the published ablation, plausibly worth more than the
entire TS-SatFire ingestion effort. **Do them first, on your existing mNDWS data,** and
you'll have a much better v3.5 baseline to judge whether TS-SatFire actually helped.

That's the key process change: **you currently have no way to attribute a v4
improvement to any specific cause**, because data + channels + losses + sampler all
change at once. Sequence them.

---

## 5. Repo and infrastructure (the mono-inspired part)

From `mono`, the transferable pattern is *every rule is executable*:

1. **`pyproject.toml` + `uv.lock`** — you have no ML dependency manifest at all today.
2. **Channel-schema drift guard** — one check asserting `config.v4.yaml`,
   `dataset.py`, and `services/tilesvc/model_config.default.json` agree on channel
   order/count, and that protected channels are in the dropout exclude list. This is
   your `migration-drop-guard` analogue, and it guards the exact class of bug that
   produced the v3 failure.
3. **`mise` tasks** — `mise ingest` / `train` / `eval` / `check`, replacing `run_v4.sh`.
4. **Experiment registry** — every run writes a model card (config hash, data hash,
   AP per fold, calibration, git sha). Non-negotiable if you're comparing v3/v4/v5.
5. **Eval regression gate in CI** — already Phase 7 of your gameplan.

---

## 6. Suggested roadmap

**Phase A — Protocol fixes (3 days, no new data).** Items 1–4 above on existing mNDWS.
Establish a v3.5 baseline with AP + LOYO-CV. *Deliverable: a trustworthy baseline and
a number the field can compare to.*

**Phase B — Reproduce SOTA (1 week).** UTAE(Res18) on WSTS, target ~0.47 AP. Validates
your harness end-to-end. *Deliverable: proof the pipeline is correct.*

**Phase C — v5 (2–3 weeks).** TS-SatFire ingestion (Phase 1 of current plan) + your
three fire-specific contributions, ablated individually. *Deliverable: a defensible
claim about what actually fixed Palisades.*

**Phase D — Product (parallel).** Time-to-arrival, uncertainty bands, OOD confidence
signal.

**Phase E — Research bet.** Probabilistic (flow matching) or physics-hybrid (neural CA).

Phase A is the one to start today — it's cheap, it's on data you already have, and it
determines whether the expensive phases are measuring anything real.

---

## Sources

- Lahrichi, Bova, Johnson, Malof — [Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark](https://arxiv.org/abs/2502.12003) (WACV 2026) — *the key paper*
- Gerard et al. — [WildfireSpreadTS benchmark](https://arxiv.org/abs/2502.12003) (via above)
- Huot et al. — [Next Day Wildfire Spread](https://arxiv.org/pdf/2112.02447)
- [TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset](https://www.nature.com/articles/s41597-025-06271-3) — *your v4 dataset*
- [Neural-Parameterized Cellular Automata for Wildfire Spread](https://arxiv.org/abs/2606.11676)
- [Physics-informed neural networks for parameter learning of wildfire spreading](https://arxiv.org/pdf/2406.14591)
- [Probabilistic Forecasting of Localized Wildfire Spread Based on Conditional Flow Matching](https://arxiv.org/pdf/2603.26975)
- [Trending and emerging prospects of physics-based and ML-based wildfire spread models](https://link.springer.com/article/10.1007/s11676-024-01783-x)
- [ML/DL for Wildfire Prediction: Systematic Review 2020–2025](https://doi.org/10.3390/fire9050204)
- [Google FireBench](https://research.google/blog/firebench-using-high-performance-computing-to-advance-machine-learning-and-wildfire-research/) · [Google Wildfire Simulation](https://sites.research.google/gr/wildfires/fire-simulation/)
- [Technosylva + Pano AI partnership](https://technosylva.com/news/technosylva-and-pano-ai-announce-partnership-to-deliver-unified-predictive-real-time-wildfire-intelligence-for-utilities-and-fire-agencies/)
- [BCWildfire benchmark](https://arxiv.org/html/2511.17597) · [FireSentry](https://arxiv.org/pdf/2512.03369)
