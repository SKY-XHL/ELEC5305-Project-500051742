# Label-Efficient Environmental Sound Classification

- **Course:** ELEC5305 Acoustics, Speech and Signal Processing
- **Student:** Haoliang Xiong
- **Student ID:** 500051742
- **Project site:** https://sky-xhl.github.io/elec5305-project-500051742/

## Research question

How does environmental-sound classification performance scale with the number of labelled examples when using classical MFCC and Log-Mel representations versus a frozen pretrained audio-language representation?

The project uses ESC-50 to study the relationship between representation choice, classification performance, labelled-data requirements, and computational cost. MFCC and Log-Mel provide controlled classical baselines. A frozen Microsoft CLAP representation will be added as the modern audio-foundation-model comparison.

## Current status

This repository reports work completed for Project Feedback Two.

### Completed

- Validation of the complete ESC-50 dataset: 2,000 five-second clips, 50 balanced classes, and five official folds.
- Leakage-safe MFCC and Log-Mel extraction with cached fixed-length feature vectors.
- Full-data RBF-SVM baseline across all five official test folds.
- Label-efficiency experiments with 1, 2, 5, 10, 20, and all available labelled examples per class.
- Five repeated random selections for each non-full label budget.
- A common standardised linear SVM probe for MFCC and Log-Mel.
- Accuracy, macro-F1, confusion matrices, training time, inference time, and reproducible sample manifests.

### In progress

- Frozen Microsoft CLAP audio-embedding extraction.
- Zero-shot CLAP evaluation with fixed text-prompt templates.
- CLAP label-efficiency experiments using the same folds, samples, and linear probe.
- Comparative class-level and ESC-50 coarse-category analysis.

No CLAP performance result is claimed in the current preliminary results.

## Dataset and evaluation protocol

[ESC-50](https://github.com/karolpiczak/ESC-50) contains 2,000 mono recordings sampled at 44.1 kHz. Each of its 50 classes contains 40 recordings. The supplied five source-aware folds are preserved throughout the project.

For each outer experiment, one official fold is held out for testing and the remaining four folds form the candidate training set. Standardisation is fitted only on the selected training data. The test fold remains fixed while the label budget and sampled training recordings change.

### Classical representations

- **MFCC:** 40 coefficients summarised by temporal mean and standard deviation, producing an 80-dimensional vector.
- **Log-Mel:** 128 Mel bands summarised by temporal mean and standard deviation, producing a 256-dimensional vector.

### Experiment 0: full-data classical baseline

MFCC and Log-Mel are evaluated with the same RBF-SVM pipeline. Hyperparameters are selected inside the outer training data using the remaining official fold identities as a predefined inner validation split.

| Representation | Accuracy, mean ± SD | Macro-F1, mean ± SD |
|---|---:|---:|
| MFCC | 0.4910 ± 0.0232 | 0.4711 ± 0.0207 |
| Log-Mel | 0.4760 ± 0.0197 | 0.4623 ± 0.0180 |

![Classical full-data baseline](results/figures/classical_metric_comparison.png)

### Classical label-efficiency experiment

The primary label-efficiency framework uses the same linear SVM probe for both representations. MFCC and Log-Mel use exactly the same sampled recordings for every test fold, label budget, and seed.

| Labels per class | MFCC accuracy | Log-Mel accuracy | MFCC macro-F1 | Log-Mel macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.1003 | 0.1191 | 0.0767 | 0.0967 |
| 2 | 0.1365 | 0.1684 | 0.1170 | 0.1443 |
| 5 | 0.1997 | 0.2250 | 0.1872 | 0.2071 |
| 10 | 0.2542 | 0.2822 | 0.2421 | 0.2650 |
| 20 | 0.3236 | 0.3187 | 0.3081 | 0.3011 |
| Full (32) | 0.3680 | 0.3505 | 0.3521 | 0.3367 |

The preliminary classical result suggests that Log-Mel is more label-efficient from 1 to 10 examples per class, while MFCC slightly exceeds Log-Mel at 20 examples and with the complete outer-fold training set. These are linear-probe results and should not be directly equated with the RBF-SVM baseline above.

![Classical label-efficiency learning curves](results/figures/classical_label_efficiency.png)

## Repository structure

```text
.
├── Assignment1_Project_Feedback_one/
├── Assignment2_Audio_Processing_Report_One/
├── src/
│   ├── validate_dataset.py
│   ├── extract_classical_features.py
│   ├── evaluate_classical_baseline.py
│   └── evaluate_label_efficiency.py
├── results/
│   ├── figures/
│   ├── classical_summary.csv
│   ├── label_efficiency_summary.csv
│   └── supporting CSV and JSON records
├── config.yaml
└── requirements.txt
```

The audio dataset, virtual environments, extracted feature caches, and model files are intentionally excluded from Git.

## Reproduction

Python 3.11 is recommended.

```powershell
git clone https://github.com/SKY-XHL/elec5305-project-500051742.git
cd elec5305-project-500051742
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50
```

Run the classical pipeline from the repository root:

```powershell
python src/validate_dataset.py
python src/extract_classical_features.py
python src/evaluate_classical_baseline.py
python src/evaluate_label_efficiency.py
```

Feature extraction processes all 2,000 audio files and can take substantially longer than the cached classification experiments.

## Key output files

- `results/dataset_validation.json`: dataset-integrity checks.
- `results/feature_extraction_summary.json`: successful and failed extraction counts and combined extraction time.
- `results/classical_fold_metrics.csv`: fold-level RBF-SVM baseline results.
- `results/classical_summary.csv`: five-fold RBF-SVM summary.
- `results/label_efficiency_raw.csv`: all fold, budget, seed, and representation results.
- `results/label_efficiency_samples.csv`: exact sampled files for reproducibility.
- `results/label_efficiency_summary.csv`: aggregated learning-curve values.
- `results/figures/`: comparison, learning-curve, and confusion-matrix figures.

## Limitations of the current checkpoint

- The current learning curves cover only MFCC and Log-Mel; frozen CLAP is not yet integrated.
- The recorded classical extraction time covers MFCC and Log-Mel extraction together and is not an individual representation-cost comparison.
- Per-clip decibel conversion uses each clip's maximum as the reference, so absolute loudness is not retained.
- The current confusion matrices support preliminary classical analysis; comparative zero-shot and few-shot CLAP analysis remains future work.

## Next milestones

1. Reproduce the official Microsoft CLAP ESC-50 zero-shot example.
2. Cache one frozen CLAP audio embedding for every ESC-50 clip.
3. Add CLAP to the common linear-probe label-efficiency evaluation.
4. Compare zero-shot and few-shot CLAP and analyse prompt sensitivity.
5. Analyse class-level and coarse-category label efficiency.
6. Compare performance, representation-extraction cost, probe cost, and storage.

## References

1. K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” *Proceedings of ACM Multimedia*, pp. 1015–1018, 2015. https://doi.org/10.1145/2733373.2806390
2. S. Chu, S. Narayanan, and C.-C. J. Kuo, “Environmental Sound Recognition With Time-Frequency Audio Features,” *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 17, no. 6, pp. 1142–1158, 2009. https://doi.org/10.1109/TASL.2009.2017438
3. K. J. Piczak, “Environmental Sound Classification with Convolutional Neural Networks,” *IEEE MLSP*, pp. 1–6, 2015. https://doi.org/10.1109/MLSP.2015.7324337
4. S. Chachada and C.-C. J. Kuo, “Environmental sound recognition: a survey,” *APSIPA Transactions on Signal and Information Processing*, vol. 3, e14, 2014. https://doi.org/10.1017/ATSIP.2014.12
5. B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang, “CLAP Learning Audio Concepts From Natural Language Supervision,” *ICASSP*, 2023. https://arxiv.org/abs/2206.04769
6. J. Turian et al., “HEAR: Holistic Evaluation of Audio Representations,” *NeurIPS Datasets and Benchmarks*, 2021. https://arxiv.org/abs/2203.03022
