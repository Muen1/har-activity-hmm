# Human Activity Recognition with a Hidden Markov Model

**Formative 2 — Hidden Markov Models**
Individual submission.

## Overview

This project builds a Hidden Markov Model (HMM) that infers a person's
activity — **standing, walking, jumping, or still** — from smartphone
accelerometer and gyroscope data. Motion data was recorded solo using the
Sensor Logger app on an iPhone 14 Pro Max (~100 Hz sampling rate for both
sensors), 13 trials per activity (52 trials total, 5–10 seconds each).

The pipeline:
1. **`src/prepare_dataset.py`** — converts the raw Sensor Logger export
   zips into clean, labelled, per-trial CSVs (`data/processed/`).
2. **`notebooks/HAR_HMM_Pipeline.ipynb`** — feature extraction (time +
   frequency domain), a `GaussianHMM` (hmmlearn) trained with Baum–Welch,
   Viterbi decoding, and evaluation on held-out, unseen trials.
3. **`report/`** — the written report (background, methods, results,
   discussion) exported as a PDF.

## Repository structure

```
.
├── .gitignore
├── README.md
├── data/
│   ├── zips_raw/                  # raw Sensor Logger exports (gitignored, local only)
│   └── processed/                 # 52 clean, labelled per-trial CSVs (committed)
│       ├── standing_trial01.csv ... standing_trial13.csv
│       ├── walking_trial01.csv  ... walking_trial13.csv
│       ├── jumping_trial01.csv  ... jumping_trial13.csv
│       ├── still_trial01.csv    ... still_trial13.csv
│       └── _dataset_summary.csv # per-trial duration / sample-rate summary
├── notebooks/
│   └── HAR_HMM_Pipeline.ipynb     # full pipeline, written/run in Google Colab
├── report/
│   └── HAR_HMM_Report.pdf         # 4-5 page written report
└── src/
    └── prepare_dataset.py         # raw zips -> clean labelled CSVs (run locally)
```

## Dataset

| Activity | Trials | Total duration | Notes |
|---|---|---|---|
| standing | 13 | ~107 s | phone handheld at waist level |
| walking  | 13 | ~119 s | consistent pace |
| jumping  | 13 | ~114 s | continuous jumps |
| still    | 13 | ~113 s | phone flat on a surface |

Single device, single sampling rate (100 Hz) throughout — confirmed per
trial in `data/processed/_dataset_summary.csv` — so no cross-device
sample-rate harmonization was needed.

## How to reproduce

### 1. Raw data → labelled CSVs (local, VS Code / terminal)

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install pandas

python src/prepare_dataset.py --input data/zips_raw --output data/processed
```

This regenerates every CSV in `data/processed/` from the raw zips and
prints a per-trial summary (row count, duration, sample rate).

### 2. Notebook (Google Colab)

1. Upload the `data/processed/` folder to your Google Drive, e.g. to
   `MyDrive/HAR_HMM_Data/processed/`.
2. Open `notebooks/HAR_HMM_Pipeline.ipynb` in Colab (upload it directly, or
   open it from GitHub via Colab's "GitHub" tab — either way, no `git
   clone` runs *inside* the notebook itself).
3. Run all cells top to bottom. The notebook mounts Google Drive, installs
   `hmmlearn`, and walks through feature extraction, training, decoding,
   and evaluation, with a markdown explanation before each step.
4. Download the finished notebook (`File > Download > .ipynb`) back into
   `notebooks/` in your local clone, then commit it.

## Results summary

On held-out, unseen test trials (trials 11–13 of every activity), the
trained model reaches strong separation between activities, with **still**
the most reliably distinguished class (its low-motion signal is an order of
magnitude flatter than any handheld activity) and confusions concentrated
between the higher-motion activities on harder train/test splits. Full
metrics, confusion matrices, and the transition/emission visualizations are
in the notebook and the report.

## Notes on the assignment's collaboration rubric criterion

This is an individual submission — there is no group task-allocation table
because there is no group. The full commit history under this single
account is the record of all work done.
