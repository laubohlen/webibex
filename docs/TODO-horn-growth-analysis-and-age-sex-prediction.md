# TODO — Horn-growth chronology analysis + age/sex prediction from photos

Raised 2026-10-23. Status: **idea stage, nothing started**. Informal. Facts below come from the 2026-10-23
session; "(summary)" marks claims I only saw in AI-generated search summaries and did not check in a primary source.

## Idea

1. Apply the published horn-growth methods (annual increments, cross-animal "common signal") at scale to the
   ~4000 photos, as far as the data allows.
2. Add data the photos lack: predict **age and sex** from the photo, then re-run the analysis with those
   predictions as extra (uncertain) covariates.

The professor's "basin analysis" wording maps to **sclerochronology / dendro-style chronology building**
("basin analysis" is the geology term).

## What the papers say (read in full: PDFs are in `~/workspace/agents_writer/tmp/webibex/`, untracked)

- Age dominates: 50% of variance, ages 1-6 (Giacometti 2002, J. Mammalogy 83:245-251). Growth drops from ~age 6-7.
- Climate signal is real but regional: March-May temperature (Swiss Alps: Giacometti 2002; Buntgen 2014, 8,043
  males, r = 0.77 between colonies); spring precipitation + winter minimum temperature (Gran Paradiso: von
  Hardenberg 2004); no weather effect in Iberian ibex (Carvalho 2020, habitat/conifer cover instead).
- Individual identity explains 22-23% of variance (Bergeron 2008; Carvalho 2020); good years follow good years
  within an animal (autocorrelation 0.525).
- The outermost (first) increment is always dropped (abrasion). von Hardenberg 2004: the last 2 years before
  death are also shorter, independent of age.
- Year-to-year signal is small: about +-5 mm on a 76.6 mm mean (Giacometti 2002 Table 2).
- Three published ways to remove age: regional curve standardisation (Buntgen 2014), mixed model with animal
  random effect + calendar-year fixed effect (Bergeron, von Hardenberg, Carvalho), age-specific Z-scores
  (Buntgen 2018).
- All of it is **males only**, measured on harvested/dead animals with calipers or tape. Photos of live animals:
  Brambilla & Canedoli 2013 (J. Mountain Ecology 9:35-46), scaled against a hand-measured 4th annulus
  (slope 0.964, R2 0.973, N = 654 annuli from 24 animals; some points off by 10-20 mm, my reading of Fig. 3).

## Phase 0 — inventory (do first, cheap)

- [ ] How many photos / distinct animals; how many animals with known cohort and sex (notes say 105 in `animals.json`).
- [ ] Do real photos keep their EXIF date after upload? (the test image was named `..._noexifdata.jpg`)
- [ ] Do the animals wear ear tags (shortcut risk for the classifier)? Which population are they (matters for
      which climate variables apply)?
- [ ] Re-read the deferred "age/zone/muzzle recognition" work (`docs/security-remediation-plan.md:443`) so nothing is redone.
- [ ] Cluster near-duplicates (embedding similarity or perceptual hash); write down the cluster sizes.

## Phase 1 — measurement and baseline analysis

- [ ] Measure annulus widths from the aligned chips. No scale in mm: start with **relative** growth (ratios to a
      reference annulus), as in Brambilla's method. Measurement error must be well below the +-5 mm signal;
      average over many animals.
- [ ] Exclude the outermost increment; keep side (L/R) handling consistent with the chip code.
- [ ] Fit the mixed model (animal random effect, calendar-year fixed, age polynomial) on known-age animals only.

## Phase 2 — age/sex classifier

- [ ] YOLO crop of the whole animal, plus a separate high-resolution horn crop for age (annuli need detail).
- [ ] Baseline: frozen DINOv2 features + linear/kNN head (strong cheap baseline for small label sets, per a
      2026 marine study (summary)); also try the existing identity-trained chip embeddings.
- [ ] Age as ordered classes (CORAL, Cao et al. 2020); report balanced accuracy, MAE in years, share within +-1 year.
- [ ] **Grouped** CV only: `StratifiedGroupKFold`, groups = animal + similarity cluster/session; repeated, with
      intervals. Plain K-Fold / shuffle-split leak near-twins (random splits overestimated >100% in
      SeaTurtleID2022 (summary)).
- [ ] Near-duplicates: cap cluster weight in training instead of augmenting them. Augmentation only as an
      ablation, and only nuisance variation (light, blur, JPEG); no scale/aspect changes. Own earlier result:
      ColorJitter+GaussNoise 75.33% vs 92.32% mAP (re-ID task).
- [ ] Calibrate the probabilities (temperature scaling, reliability plot) and let the model abstain.
- [ ] Audit: random sample labelled by an expert independent of the model; ~385 samples for +-5% on one overall
      accuracy figure, more for per-class figures; stratify by predicted class and confidence.

## Phase 3 — re-run with predictions

- [ ] Use predicted age/sex as soft labels; run the analysis twice (known only vs known + predicted) and compare.
- [ ] Check year alignment dendro-style (cross-dating): a +-1 year age error shifts every segment's calendar year
      and washes out the shared signal. My reasoning, not from a source.

## Risks

- ~100 animals with known labels is a small training base; photos of the same animal are not independent.
- Ground truth covers marked animals only; unmarked ones may differ (audit them too).
- Predicted age and measured annuli come from the same horn feature: errors are correlated.
- Females: no published method in what I read; their annuli may be less distinct.

## Open questions

- Dataset size per animal, and whether photo dates exist.
- Is reviving the age/zone/muzzle work part of what the professor wants?
- Who labels the audit set?

## Not in scope

- Production deployment of the classifier; any change to the live webibex app.
