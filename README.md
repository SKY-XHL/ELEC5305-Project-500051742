# Label-Efficient Environmental Sound Classification

ELEC5305 final project by **Haoliang Xiong (Student ID 500051742)**.

This project measures how ESC-50 classification performance scales with the number of task-specific labelled examples when audio is represented using pooled MFCCs, pooled Log-Mel features, or frozen Microsoft CLAP embeddings. All supervised comparisons use the official ESC-50 folds, shared sampled filenames, leakage-safe scaling, and a common linear-probe protocol.

## Headline results

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| CLAP zero-shot, predeclared primary prompt | 93.85% | 93.75% |
| CLAP, 5 labels per class | 96.45% | 96.37% |
| CLAP, full training data | 97.55% | 97.46% |
| Best full-data classical baseline, MFCC | 36.80% | 35.21% |

Five-shot CLAP used 84.4% fewer task-specific labels than the full supervised condition while retaining 98.87% of its accuracy. These findings apply to the ESC-50 official folds and do not establish broad out-of-domain generalisation.

## Project structure

```text
Final_Project/
├── index.html                         # Static project page
├── assets/
│   ├── styles.css                     # Page design
│   ├── script.js                      # Gallery, navigation and lightbox
│   ├── site-config.js                 # Public GitHub and video URLs
│   ├── workflow.png                   # Main methodology figure
│   ├── learning_curves.png            # Main label-efficiency figure
│   ├── confusion_analysis.png         # Main error-analysis figure
│   └── figures/                       # Additional analysis figures
├── src/                               # Experiment and report scripts
├── results/                           # Metrics, predictions and source figures
├── docs/
│   ├── ELEC5305_Final_Project_Report.pdf
│   └── ELEC5305_Final_Project_Report.docx
├── video/                             # Video notes or final local video
├── config.yaml                        # Fixed experiment configuration
├── references.bib
├── requirements.txt                   # Classical-analysis environment
└── requirements-clap-lock.txt         # Locked CLAP environment
```

The dataset, pretrained model files, virtual environments and large cached feature arrays are intentionally excluded from version control.

## Dataset

Download ESC-50 and place it at:

```text
data/ESC-50/
├── audio/
└── meta/esc50.csv
```

Expected properties: 2,000 mono WAV clips, 50 balanced classes, five official folds, 44.1 kHz sampling rate and five-second duration.

## Environments

Two Python environments were used because the classical-analysis environment and the Microsoft CLAP dependency stack require different NumPy/librosa versions.

### Classical analysis

```powershell
py -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### Microsoft CLAP

The exact successful environment is recorded in `requirements-clap-lock.txt`. On the project machine it used Microsoft CLAP 1.3.4 and CPU-only PyTorch.

```powershell
py -m venv .venv-clap
& ".\.venv-clap\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-clap\Scripts\python.exe" -m pip install -r requirements-clap-lock.txt
```

## Reproduction order

Run all commands from `Final_Project`.

```powershell
# 1. Validate the dataset
& ".\.venv\Scripts\python.exe" ".\src\validate_dataset.py"

# 2. Extract MFCC and Log-Mel representations
& ".\.venv\Scripts\python.exe" ".\src\extract_classical_features.py"

# 3. Evaluate tuned full-data classical baselines
& ".\.venv\Scripts\python.exe" ".\src\evaluate_classical_baseline.py"

# 4. Confirm the CLAP model and one-file audio path
& ".\.venv-clap\Scripts\python.exe" ".\src\smoke_test_clap.py"

# 5. Extract frozen CLAP audio embeddings
& ".\.venv-clap\Scripts\python.exe" ".\src\extract_clap_embeddings.py"

# 6. Evaluate zero-shot CLAP
& ".\.venv-clap\Scripts\python.exe" ".\src\evaluate_clap_zero_shot.py"

# 7. Run repeated label-efficiency probes
& ".\.venv\Scripts\python.exe" ".\src\evaluate_label_efficiency.py"

# 8. Generate final statistical summaries and figures
& ".\.venv\Scripts\python.exe" ".\src\analyze_final_results.py"
```

The full CLAP extraction is cached because it took approximately 1,016 seconds on CPU. `analyze_final_results.py` uses saved outputs and does not retrain a model or re-extract features.

## Static project page

The website has no external dependencies. It can be opened directly through `index.html`, but a local HTTP server gives more consistent browser behaviour:

```powershell
& ".\.venv\Scripts\python.exe" -m http.server 8000
```

Then open `http://localhost:8000/`.

Before submission, edit `assets/site-config.js` and add the public GitHub repository URL and public or unlisted video URL:

```javascript
window.PROJECT_LINKS = {
  repository: "https://github.com/your-account/your-repository",
  video: "https://your-video-link"
};
```

The corresponding buttons remain disabled until URLs are supplied, avoiding broken placeholder links.

## GitHub Pages

1. Create a GitHub repository and upload the contents of `Final_Project`.
2. In repository **Settings → Pages**, deploy from the main branch and repository root.
3. Confirm that `index.html`, the report, images and README links load correctly.
4. Add the final Pages URL to the written report.
5. Test the project and video links in a private/incognito browser window.

## Report and video

- Project website: https://sky-xhl.github.io/ELEC5305-Project-500051742/
- Final report: `docs/ELEC5305_Final_Project_Report.pdf`
- Project video: https://github.com/SKY-XHL/ELEC5305-Project-500051742/releases/tag/video-v1

The video should briefly show the research question, the processing pipeline, code running on a small example, the main learning curve, confusion reduction, limitations and the final conclusion. The complete 2,000-clip CLAP extraction should not be rerun during the recording.
