# Session-to-Session Drift in Motor Imagery EEG (BNCI2014-004)

ROBT613 Midterm Project

## Team

| Member | Role (leads) | Reviews |
|---|---|---|
| Sultanbek Myrzakhmet | Data and provenance lead | Evaluation |
| Aruzhan Nurmanova | Signal processing lead | Data and provenance |
| Daniela Gabdullina | Decoding lead | Signal processing |
| Aidar Toktargazin | Evaluation and reporting lead | Decoding |

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the dated contribution log.

## Dataset

BNCI2014-004 (BCI Competition IV 2b, Graz). 2-class motor imagery (left/right hand), 9 subjects, 5 sessions per subject on
different days (S1–S2 screening, S3–S5 with smiley feedback), 3 bipolar EEG channels (C3, Cz, C4) plus 3 EOG channels,
250 Hz. The original `.mat` files are read directly; MOABB is used only to download them.

## Research question and hypotheses (fixed in advance)

Does a motor-imagery decoder trained on an earlier day perform worse on a new day than a decoder trained on that same day?
Drift cost = within-session AUC − cross-session AUC, with the same number of training trials.

| | Hypothesis | Test |
|---|---|---|
| **H1a** (primary) | Train S1, test S2: cross-session AUC lower than within-session CV on S2 | one-sided Wilcoxon over 9 subjects, α = 0.05 |
| **H1b** (secondary) | Same in the feedback paradigm: train S3, test S4 and S5 | same |
| **H2** | Label-free fixes reduce the drift cost: **C** baseline-relative power, **D** per-session z-scoring | one-sided Wilcoxon per fix, Holm-corrected |

## Pipeline

1. **Data audit** — trial counts, class balance, expert artifact flags, pre-trial EOG calibration recording
2. **Signal processing** — FIR band-pass 1–40 Hz; EOG regression fitted on the calibration recording
3. **Artifact handling** — expert-flagged and >5 robust-SD variance-outlier trials left out of *training* only; all trials scored
4. **Neurophysiological check** — mu/beta ERD over C3/C4 and time–frequency signed r²
5. **Features and classifier** — Welch log band power (mu 8–12, beta 13–30 Hz × C3, Cz, C4), scaler + logistic regression
6. **Honest evaluation** — leave-one-run-out within-session CV vs. cross-session decoder, matched training size
   (20 stratified subsamples), AUC scored run by run on identical trials; shuffled-label check; Mann–Whitney chance test
7. **Hypothesis tests** — one-sided Wilcoxon across subjects, Holm correction for H2, six sensitivity analyses

## Results

| | within AUC | cross AUC | median drift | p |
|---|---|---|---|---|
| H1a S1→S2 | 0.677 | 0.656 | −0.002 | 0.545 |
| H1b S3→S4/S5 | 0.872 | 0.869 | +0.003 | 0.082 |

No measurable drift in class separation; a small threshold shift in feedback sessions (balanced-accuracy drift +0.011,
p = 0.014). Neither fix is supported (C lowers cross-session AUC by 0.066, p = 0.004; D changes nothing). The conclusion
holds in all six sensitivity analyses. Full write-up: [`report/report.pdf`](report/report.pdf); slides:
[`slides/ROBT613_slides.pptx`](slides/ROBT613_slides.pptx).

## Reproducibility

- Main notebook: `notebooks/main.ipynb` (runs top to bottom; installs `mne`, `moabb`, `nilearn` if missing)
- Reusable code: `src/` (`data.py`, `preprocess.py`, `features_models.py`, `evaluation.py`)
- Random seed: 42 (every comparison has its own seeded stream)
- Subject IDs used: all 9 (1–9), all 5 sessions
- Package versions: see `requirements.txt`
- Colab link: TODO

## Repository layout

```
notebooks/main.ipynb        full analysis with outputs
src/data.py                 settings, .mat loading, data audit
src/preprocess.py           filtering, EOG regression, epoching, trial QC, ERD
src/features_models.py      log band-power features (A/C/D), classifier
src/evaluation.py           within vs cross evaluation, H1/H2 tests
figures/                    figures exported from the notebook
report/                     LaTeX report (main.tex) and compiled report.pdf
slides/                     presentation
CONTRIBUTIONS.md
requirements.txt
```

## Citations

Methods follow notebooks 00–06 of the [ROBT613 BCI Workshop](https://berdakh.github.io/ROBT613-BCI-Workshop/);
`.mat` loading follows the course lecture *Motor-Imagery BCI decoding with CSP and LDA*. Code reused from the workshop
notebooks is cited in comments where it is used.

## AI tools used

- Claude (Anthropic): drafting the slides and the LaTeX report from the notebook results, refactoring the notebook code
  into the `src/` modules, and organising this repository.
