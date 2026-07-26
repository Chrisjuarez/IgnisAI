# Pretrained models & code you can actually download

Survey of what exists off-the-shelf for wildfire spread prediction, what each
thing is genuinely good for, and where it does *not* apply. The last column is
the one that matters — several of these look relevant and are not.

---

## 1. The headline: WSTS+ SOTA checkpoints are public

**https://huggingface.co/saadlahrichi/WSTSPlus/tree/main/trained_model_weights** — 6.3 GB

Published by Lahrichi et al. alongside the WACV 2026 paper that the whole
`v5-research-informed-redesign.md` analysis is based on. Three variants:

| Directory | What it is | Reported AP (WSTS, 12-fold LOYO) |
| --- | --- | --- |
| `Res18UTAE_T5/` | **Overall SOTA** — UTAE + pretrained ResNet-18 encoder, T=5 | **0.478** |
| `Res18Unet_T5/` | ResNet-18 U-Net, data-level fusion, T=5 | 0.472 |
| `Res18Unet_T1/` | ResNet-18 U-Net, single-day | 0.468 |

Code: **https://github.com/slahrichi/WildfireSpreadTS** (Lahrichi's fork)
Original benchmark code: **https://github.com/SebastianGer/WildfireSpreadTS**
Dataset (HDF5): **https://zenodo.org/records/8006177**
Worked example consuming these weights: **https://github.com/jonasvilhofunk/WildfireUQ-FCER**

### Why this matters more than it first looks

Phase B of the redesign plan was "reproduce UTAE(Res18) on WSTS, target ~0.47 AP,
about a week." **That week just became a download.** You can skip reproduction and
go straight to the thing reproduction was meant to give you: a trustworthy
reference model whose number is comparable to published literature.

### The catch — it is not a drop-in for tilesvc

| | WSTS / Res18UTAE | IgnisAI |
| --- | --- | --- |
| Channels | 23 | 13 dynamic / 15 static |
| Resolution | 375 m | 500 m |
| Tile | 128×128 | 64×64 |
| Target | next-day full mask | delta (new burn only) |
| CRS/grid | dataset-native | EPSG:5070 tilesvc grid |

You cannot `load_state_dict` this into `ConvLSTMUNet`. What you *can* do, in
increasing order of effort:

1. **Run it as-is on WSTS** to get a calibrated sense of what "good" looks like,
   and to sanity-check your own eval code against a known-good number.
2. **Evaluate it on your 5 OOD presets** (Palisades, Eaton, Camp, Dixie, Caldor)
   by building WSTS-format inputs for those events. This directly answers "is
   the published SOTA any good on Santa Ana events?" — which is *your actual
   research question*, and nobody has published the answer.
3. **Fine-tune it** on TS-SatFire + your Santa Ana corpus. Transfer from a
   0.478-AP starting point is a fundamentally different proposition from
   training ConvLSTM from scratch.

Option 2 is the high-value, low-cost move: no training at all, and if the SOTA
model *also* fails on Palisades, that is a publishable finding and a much
stronger justification for your wind-alignment and Santa-Ana-sampler work.

---

## 2. Prithvi (NASA/IBM) — useful, but not for this task

- **https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars**
- **https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar**
- Fine-tune tooling: `pip install terratorch`; https://github.com/NASA-IMPACT/hls-foundation-os

Geospatial foundation model trained on Harmonized Landsat/Sentinel-2, fine-tuned
for burn-scar segmentation (IoU 0.73 on the burn class, 0.96 overall).

**Be precise about what this does.** Burn-scar segmentation maps *where fire has
already burned* from post-hoc imagery. That is a different problem from
next-day spread *prediction*. It is not a fire-spread model and will not
forecast anything.

Where it could genuinely help IgnisAI:

- **Encoder initialization.** The Prithvi backbone has learned strong
  multispectral earth-observation representations. Using it as a frozen or
  fine-tuned encoder is a plausible upgrade over ImageNet weights — though
  note the WSTS+ ablation found pretraining worth ~0% on their Veg subset, so
  do not assume a large win.
- **Ground-truth generation.** Prithvi burn-scar output could label observed
  perimeters for events lacking official CAL FIRE / WFIGS polygons, expanding
  your `data/perimeters/` OOD set.

The second use is the more immediately valuable one and is underrated.

---

## 3. Encoder weights worth knowing about

| Source | What | Relevance |
| --- | --- | --- |
| **ImageNet ResNet-18** via `segmentation_models_pytorch` | Generic RGB pretraining | This is literally what the WSTS+ SOTA uses. `pip install segmentation-models-pytorch`, `encoder_weights="imagenet"`. Cheapest possible upgrade over your from-scratch ConvLSTM. |
| **PASTIS UTAE weights** | UTAE pretrained on crop-type time-series segmentation | What Lahrichi used to initialize UTAE (4th-fold checkpoint). Proven transfer path *into the exact architecture* — temporal-attention weights, not just spatial. |
| **TorchGeo** (`pip install torchgeo`) | SSL4EO-S12 MoCo/DINO ResNet50, DOFA, SeCo, MAE | First library supporting multispectral-sensor pretraining. Better prior than ImageNet for satellite bands. Supports timm encoders + SMP decoders, so it composes with the above. |

---

## 4. Recommended sequence

**Now (no GPU, ~1 day):** download `Res18UTAE_T5`, stand up the WSTS dataloader
from `slahrichi/WildfireSpreadTS`, confirm you reproduce ~0.478 AP. This
validates your entire eval harness against a known-good number — the same
motivation as Phase B, at a fraction of the cost.

**Next (~2 days):** build WSTS-format inputs for the 5 OOD presets and score the
public SOTA on them. Answers whether the field's best model handles Santa Ana
events. Either result is valuable: if it fails, your problem is real and
under-studied; if it succeeds, you have a much better starting checkpoint than
anything you would train.

**Then:** fine-tune from `Res18UTAE_T5` rather than training ConvLSTM from
scratch. Keep the ConvLSTM line alive only as the deployed v3/v4 fallback.

**Cheap parallel win:** swap an ImageNet-pretrained ResNet-18 encoder into the
current architecture. Small change, and it is one of the ingredients behind the
paper's +37%.

---

## 5. What this does *not* solve

- **The 22.6% split leakage** in `mNDWS_500m_T3` — a pretrained model does not
  fix a contaminated evaluation. Group-aware split still required.
- **Channel/grid mismatch with tilesvc.** Serving still expects 64×64 @ 500 m on
  EPSG:5070. Adopting a WSTS-native model means either resampling at serve time
  or re-tiling production.
- **The Santa Ana question.** No public model was trained on a Santa-Ana-rich
  corpus. That gap is still yours to fill — which is the point.

---

## Sources

- [WSTS+ weights (HuggingFace)](https://huggingface.co/saadlahrichi/WSTSPlus/tree/main/trained_model_weights) · [paper](https://arxiv.org/abs/2502.12003) · [code](https://github.com/slahrichi/WildfireSpreadTS)
- [WildfireSpreadTS benchmark](https://github.com/SebastianGer/WildfireSpreadTS) · [dataset (Zenodo)](https://zenodo.org/records/8006177)
- [WildfireUQ-FCER — worked example using these weights](https://github.com/jonasvilhofunk/WildfireUQ-FCER)
- [Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) · [Prithvi-100M-burn-scar](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar) · [fine-tuning repo](https://github.com/NASA-IMPACT/hls-foundation-os)
- [TorchGeo](https://github.com/torchgeo/torchgeo) · [pretrained model docs](https://docs.torchgeo.org/en/latest/api/models.html)
