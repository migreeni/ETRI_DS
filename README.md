다음은 첨부된 `f5c46c76-1024-4de3-98d6-2005b6c85251.png` 파일(수면 품질 및 상태 예측 프로젝트)에 대한 README 파일 예시입니다:

---

# README

## Project Title

**Sleep Quality and Condition Prediction using Lifelog Data**

## Overview

This project aims to predict six sleep-related metrics using daily lifelog data collected from multimodal sensors (smartphone, smartwatch, sleep mat) and self-reported surveys. The task was conducted as part of the [Human Understanding AI Paper Challenge 2025](https://aifactory.space/task/2524/overview), which evaluates models on their ability to infer both subjective (questionnaire-based) and objective (sensor-based) indicators of sleep quality.

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

* Wonjun Choi (Korea University)
* COSE471: Data Science, Term Project

## References

* National Sleep Foundation Guidelines
* Withings Sleep Analyzer
* [ETRI Lifelog Dataset 2020](https://nanum.etri.re.kr/share/schung/ETRILifelogDataset2020)
* Oh et al., “Sensor-Based Multi-Label Dataset Analysis Challenge”, ICTC 2024
* [Dacon Competition Overview](https://dacon.io/competitions/official/236240/overview)

---

필요시 영어 버전 외에 한국어 버전도 작성 가능합니다. 추가 항목이나 포맷 수정이 필요하면 알려주세요.
