# Human Activity Recognition with a Hidden Markov Model

**Formative 2 — Hidden Markov Models**

## Overview

Human Activity Recognition (HAR) is important because it is used to figure out a person’s physical activities without any human assistance. However, HAR faces a major problem which is that smartphones often capture messy data that is inconsistent and requires heavy computers to process[1]. In this project, the goal is to track fitness and activity by using smartphone accelerometer and gyroscope measurements to recognise walking, standing, jumping, and still when there is no movement. I used a Hidden Markov Model because it captures the activity as a sequence of hidden states while accounting for temporal dependencies between consecutive observations, allowing noisy sensor predictions to be smoothed into more consistent activity classifications over time. This temporal modeling makes HMMs well suited for improving the reliability of smartphone-based fitness monitoring systems, where activities naturally occur in continuous sequences rather than as isolated events.

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
│   ├── zips_raw/                  
│   └── processed/                
│       ├── standing_trial01.csv ... standing_trial13.csv
│       ├── walking_trial01.csv  ... walking_trial13.csv
│       ├── jumping_trial01.csv  ... jumping_trial13.csv
│       ├── still_trial01.csv    ... still_trial13.csv
│       └── _dataset_summary.csv 
├── notebook/
│   └── HAR_HMM_Pipeline.ipynb     
├── report/
│   └── HAR_HMM_Report.pdf        
└── src/
    └── prepare_dataset.py         
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
source .venv/bin/activate        
pip install pandas

python src/prepare_dataset.py --input data/zips_raw --output data/processed
```

This regenerates every CSV in `data/processed/` from the raw zips and
prints a per-trial summary (row count, duration, sample rate).

### 2. Notebook (Google Colab)

1. Upload the `data/processed/` folder to your Google Drive, e.g. to
   `MyDrive/HAR_HMM_Data/processed/`.
2. Open `notebooks/HAR_HMM_Pipeline.ipynb` in Colab 
3. Run all cells top to bottom. The notebook mounts Google Drive, installs
   `hmmlearn`, and walks through feature extraction, training, decoding,
   and evaluation, with a markdown explanation before each step.
4. Download the finished notebook (`File > Download > .ipynb`) back into
   `notebooks/` in your local clone, then commit it.

## Results summary

I recorded 13 trials for each activity. Trials 1-10 were used for training the model while trials 11-13 were used to test the performance of the model. Therefore, the dataset was split by a whole trial to prevent data leakage. This ensured overlapping windows from the same recording did not appear in both training and test sets. This resulted in a training set of 632 windows from 40 trials and a test set of 200 windows from 12 unseen trials. This shows the ability of the Hidden Markov Model to generalize new recordings.
