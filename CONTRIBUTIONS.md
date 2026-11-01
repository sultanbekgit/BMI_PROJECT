# Contribution Log

Project: Session-to-session drift on BNCI2014-004


## Roles

| Member | Leads | Reviews |
|---|---|---|
| Sultanbek Myrzakhmet | Data and provenance: loading, data audit, event coding, subject selection | Evaluation |
| Aruzhan Nurmanova | Signal processing: filtering, artifact handling, quality control | Data and provenance |
| Daniela Gabdullina | Decoding: features, classifier pipeline, cross-validation | Signal processing |
| Aidar Toktargazin | Evaluation and reporting: chance levels, statistics, figures, report | Decoding |

## Log

| Date | Member | Task | Commit / PR link |
|---|---|---|---|
| 2026-10-28 | Sultanbek | README, contribution log and requirements template | |
| 2026-11-01 | Sultanbek | `.mat` loader, analysis settings, data audit (`src/data.py`); audit, recording-site and trial-timeline figures | |
| 2026-11-01 | Aruzhan | FIR filtering, EOG regression on the calibration recording, epoching, trial QC, ERD (`src/preprocess.py`); EOG, cleaning, ERD and time–frequency figures | |
| 2026-11-01 | Daniela | Log band-power features A/C/D, classifier, stratified size matching (`src/features_models.py`); decoder-weight, drift and feature-space figures | |
| 2026-11-01 | Aidar | Within vs cross evaluation, H1/H2 tests, Holm correction (`src/evaluation.py`); result, sensitivity and validation figures | |
| 2026-11-01 | Aidar | LaTeX report (`report/`) | |
| 2026-11-01 | Sultanbek | Full analysis notebook, slides, README and requirements update | |

AI assistance (Claude) was used to draft the slides and report and to refactor the notebook into `src/`; see README.

## Reviews

| Date | Reviewer | Reviewed | Outcome | PR link |
|---|---|---|---|---|
