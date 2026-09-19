# Income Status Prediction — Upgraded Machine Learning Project

An upgraded, portfolio-ready version of the classic Adult/Census Income classification case study.

## What is upgraded?

The original case study is expanded into a reproducible end-to-end machine-learning project:

- Robust preprocessing with separate numeric and categorical pipelines
- Train/validation/test workflow with stratification
- Multiple models: Logistic Regression, Random Forest, and HistGradientBoosting
- Cross-validation and randomized hyperparameter tuning
- Accuracy, precision, recall, F1, ROC-AUC and PR-AUC
- Confusion matrix + ROC/precision-recall curves
- Threshold analysis so the classification cutoff is not treated as fixed by default
- Permutation-based feature importance / explainability
- Subgroup performance analysis for sex, with explicit fairness caveats
- Model persistence with Joblib
- CLI training and prediction scripts
- Streamlit demo application
- Unit tests and GitHub Actions CI
- Reproducible requirements and project structure

## Project structure

```text
Income_Status_ML_Upgraded/
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── income_status_prediction_upgraded.ipynb
├── reports/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

This project uses the UCI Adult Census Income dataset through `sklearn.datasets.fetch_openml` (Adult, version 2). The dataset is downloaded at runtime rather than committed to GitHub.

Target:
- `<=50K`
- `>50K`

## Quick start

```bash
git clone <your-repository-url>
cd Income_Status_ML_Upgraded
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
python -m src.train
```

The training script saves the selected model and a metrics file under `models/` and `reports/`.

Run the demo:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest -q
```

## Notebook

Open `notebooks/income_status_prediction_upgraded.ipynb` for the full walkthrough: problem definition, EDA, preprocessing, model comparison, tuning, threshold analysis, explainability, subgroup analysis and final conclusions.

## Reproducibility

The default random seed is fixed in `src/config.py`. All major transformations are implemented inside scikit-learn pipelines to reduce train/test leakage risk.

## Responsible-use note

Income status is a sensitive socioeconomic outcome. Model performance can vary across demographic groups because the underlying data reflects historical and structural patterns. This repository is intended for learning and experimentation, not for automated decisions about lending, hiring, insurance, benefits, or other high-impact decisions.

## Attribution

The project is inspired by the public machine-learning tutorial/case study linked by the requester. The upgraded implementation and project scaffolding here are newly authored for this repository.
