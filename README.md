## Project Title

**Sleep Quality and Condition Prediction using Lifelog Data**

## Overview

This project aims to predict six sleep-related metrics using daily lifelog data collected from multimodal sensors (smartphone, smartwatch, sleep mat) and self-reported surveys. The task was conducted as part of the Human Understanding AI Paper Challenge 2025, which evaluates models on their ability to infer both subjective (questionnaire-based) and objective (sensor-based) indicators of sleep quality.

## Task Description

The model is designed to predict the following six target metrics:

* **Q1**: Sleep quality (0: below individual average, 1: above)
* **Q2**: Pre-sleep fatigue (0: high, 1: low)
* **Q3**: Pre-sleep stress (0: high, 1: low)
* **S1**: Total Sleep Time (0: not recommended, 1: may be appropriate, 2: recommended)
* **S2**: Sleep Efficiency (0: inappropriate, 1: recommended)
* **S3**: Sleep Onset Latency (0: inappropriate, 1: recommended)

Each metric is predicted independently per participant, per day, using a multi-model GRU-based architecture.

## Dataset

* **Train Data**: Collected in 2020 from 22 subjects (508 days).
* **Validation / Test Data**: Collected in 2023 from 4 participants (subject\_id: 5–8).
* **Input Features**: Sensor data (accelerometer, GPS, app usage, light, heart rate, step count, ambient sound) and daily surveys.
* **Target Labels**: Derived from questionnaires and Withings Sleep Analyzer logs.

## Model Architecture

* GRU-based sequential model

  * Input: Time-series data sampled every 10 minutes (max length: 144 per day)
  * Architecture: 2 GRU layers + FC layer per target
  * Output: Multi-label prediction for Q1–Q3, S1–S3

## Training Setup

* Framework: PyTorch
* Cross-validation: 5-fold
* Metric: **Macro F1-Score**

  * Public Score: Sampled 44% of test set
  * Private Score: Full test set

## Submission Format

CSV file with the following columns:

```
subject_id, lifelog_date, Q1, Q2, Q3, S1, S2, S3
```

Each value is a prediction (0, 1, or 2 for S1 only).

## External Resources

Use of pretrained models and external datasets is allowed under:

* Open-source licenses
* Research-only terms (non-commercial)

## Dependencies

* Python 3.10+
* pandas, numpy, scikit-learn
* torch 2.1+
* tqdm, matplotlib (for analysis and logging)

## Authors

* COSE471: Data Science, Term Project

## References

* National Sleep Foundation Guidelines
* Withings Sleep Analyzer
* [ETRI Lifelog Dataset 2020](https://nanum.etri.re.kr/share/schung/ETRILifelogDataset2020)
* Oh et al., “Sensor-Based Multi-Label Dataset Analysis Challenge”, ICTC 2024

## 📁 Data Setup

Due to the large size of the dataset, please download the data files manually and place them in a local folder named `data/`.

- `ch2025_metrics_train.csv`: Training dataset  
- `ch2025_submission_sample.csv`: Test dataset (for submission format)

## 📂 Code Structure and Description

### 🧹 Preprocessing

- `preprocess_original_final.py`  
  → Preprocessing pipeline for raw/original input data  
- `preprocess_dwt_final.py`  
  → Preprocessing pipeline applying Discrete Wavelet Transform (DWT)

### 🧠 Model Execution

- `main_original.py`  
  → Training and evaluation code for individual models on original data  
- `main.py`  
  → Unified script for training models and selecting the optimal model per target metric (Q1–Q3, S1–S3)

- `run.sh`  
  → Shell script to execute `main.py` with different models and datasets

- `execution.ipynb`  
  → Jupyter Notebook for running predictions and comparing per-metric results to select best-performing models

### 📊 Model Interpretation

- `shap.py`  
  → Script for computing SHAP (Shapley Additive exPlanations) values for model interpretability

### 🤖 GRU Model

- `GRU/`  
  → Contains GRU-specific preprocessing, model training, and evaluation code for sequential prediction

### 🧪 Additional Experiments

- `other/`  
  → Contains scripts used during various experiment stages (e.g., early runs, tests, ablations)

---

## 📌 Notes

- All team members contributed to data preprocessing, predictive model development, result analysis, presentation preparation, and writing.
- Final model selection was done per target metric, leading to improved performance over single-model approaches.

---

## 🏁 How to Run

```bash
bash run.sh

