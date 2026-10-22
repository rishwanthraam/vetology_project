# Vetology Project — Canine Thorax Evaluation

## Overview

This project evaluates AI-generated predictions for canine thorax disease classification by comparing them with a manually reviewed Gold Standard.

Python is used to compare the labels and calculate confusion matrix metrics for each disease.

## Objectives

- Compare AI predictions against Gold Standard labels.
- Calculate True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN).
- Evaluate model performance using standard classification metrics.
- Export evaluation results into an Excel report.

## Technologies Used

- Python
- Pandas
- OpenPyXL
- Visual Studio Code
- Git and GitHub

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted positive cases that are actually positive |
| Recall (Sensitivity) | Proportion of actual positive cases correctly identified |
| Specificity | Proportion of actual negative cases correctly identified |

## Project Structure

```text
vetology_project/
├── confusion_matrix.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/vetology_project.git
cd vetology_project
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. Keep the required input workbooks in the appropriate local project folder.
2. Update the file paths in the Python script if necessary.
3. Run the evaluation script:

```bash
python confusion_matrix.py
```

4. Review the generated evaluation report locally.

**Note:** The input Excel files must have the expected sheet names, CaseID values, and disease label columns required by the script.

## Data Privacy

This repository is intended to contain source code and documentation only. Datasets, Excel workbooks, patient-related information, environment files, and generated reports must remain excluded from GitHub in accordance with internship data-handling requirements.

## Disclaimer

This project evaluates agreement between AI-generated labels and the supplied Gold Standard. Evaluation metrics measure classification performance and do not independently establish medical or veterinary diagnostic accuracy.
