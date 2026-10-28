# Session-to-Session Drift in Motor Imagery EEG (BNCI2014-004)

ROBT613 Midterm Project

## Team

| Member | Role (leads) | Reviews |
|---|---|---|
| Sultanbek | Data and provenance lead | Evaluation |
| Aruzhan | Signal processing lead | Data and provenance |
| Daniela | Decoding lead | Signal processing |
| Aidarkhan | Evaluation and reporting lead | Decoding |

(Roles are assigned in the order the names were listed. Change them here and in CONTRIBUTIONS.md if the team decides differently.)

## Dataset

BNCI2014-004 (BCI Competition IV 2b), loaded through MOABB. 2-class motor imagery (left/right hand), 9 subjects, 5 sessions per subject, 3 bipolar EEG channels (C3, Cz, C4) plus 3 EOG channels.

## Hypothesis

Per-session feature normalization reduces the drop in ROC AUC that occurs when a motor-imagery decoder trained on one session is tested on a later session.

## Pipeline (7 required stages)

1. Data audit
2. Signal inspection (PSD before/after filtering)
3. Artifact handling (EOG regression, rejection counts)
4. Neurophysiological check (mu/beta ERD over C3/C4)
5. Features and classifier (Welch log band power, logistic regression, one scikit-learn Pipeline)
6. Honest evaluation (GroupKFold by session, ROC AUC, permutation chance level, nested CV)
7. Hypothesis test (paired Wilcoxon across subjects)

## Reproducibility

- Main notebook: `notebooks/main.ipynb` (runs top to bottom on a fresh Colab CPU runtime)
- Random seed: TODO
- Subject IDs used: TODO
- Package versions: see `requirements.txt`
- Colab link: TODO

## Repository layout

```
notebooks/main.ipynb
src/data.py
src/preprocess.py
src/features_models.py
src/evaluation.py
figures/
report/
slides/
CONTRIBUTIONS.md
requirements.txt
```

## Citations

Code reused from the ROBT613 BCI Workshop notebooks is cited in comments where it is used.

## AI tools used

TODO: list the tools and what they were used for (debugging, API lookup, plotting).
