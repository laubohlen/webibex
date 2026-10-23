# Proposal sketch — horn-growth chronology + age/sex prediction (informal, for a chat with the professor)

Draft 2026-10-23. Details and sources: `docs/TODO-horn-growth-analysis-and-age-sex-prediction.md`.

## The idea in three lines

- Annual horn increments work like tree rings: shared year-to-year signals show up across animals. The field calls
  this sclerochronology (the "basin analysis" analogy).
- Published studies did it on harvested/dead animals (thousands of horns). We have ~4000 photos of live animals.
- Our twist: predict age and sex from the photo with a classifier, then re-run the analysis with those predictions.

## What is already known (so we don't oversell)

- Spring temperature (Swiss Alps), age (about half of the variance) and individual quality (~22%) explain a lot of
  increment length; the climate signal is only a few mm per year.
- Climate effects differ by region (Gran Paradiso: spring rain and winter minimum temperature; Iberian ibex:
  habitat, not weather).
- Photo-based annulus measurement exists (Brambilla & Canedoli 2013) but needs a hand-measured reference.
- I found no published deep-learning age-from-horn pipeline for ibex (one standard search, not exhaustive).

## What is new on our side

- Scale: far more animals/photos than a photo study has used.
- Age/sex predicted as probabilities, validated on animals with known cohort, audited by random samples.
- Honest handling of the uncertainty: run the analysis with and without the predictions and compare.

## What I'd need from the professor

- Ground truth: cohort, sex, capture date for as many animals as possible; photo dates.
- Which population/region (it decides which climate variables to test).
- Whether the animals wear ear tags.
- Someone to label a random audit sample (~400 photos).
- Any female data; every paper I read is male-only.

## Honest caveats

- ~100 labelled animals is small; photos of one animal are not independent.
- Without a mm scale we start with relative growth, not millimetres.
- The climate signal may be too small to detect with photo-measurement noise.
- Expect 4-6 weeks of part-time work for a first baseline (my rough guess, not a plan).
