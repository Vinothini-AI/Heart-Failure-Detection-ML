# ❤️ Heart Failure Detection using Machine Learning

<p align="center">
  <b>A Machine Learning project for predicting death events in heart failure patients</b>
</p>

---

## 📌 About the Project

Heart failure is a serious cardiovascular condition that requires early identification of high-risk patients.

This project uses **Machine Learning classification algorithms** to predict whether a patient is likely to experience a death event based on clinical information.

The project includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data analysis and preprocessing
- 📈 Data visualization
- 🤖 Machine Learning classification
- 📏 Model evaluation
- 🔍 Feature importance analysis
- 🔮 Patient prediction

> ⚠️ **Disclaimer:** This project is developed for educational purposes only. It is not intended to provide medical diagnosis or treatment recommendations.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Analyze the heart failure clinical records dataset.
2. Understand the important clinical features.
3. Perform Exploratory Data Analysis.
4. Visualize relationships between clinical features and death events.
5. Train multiple Machine Learning classification models.
6. Compare the performance of different models.
7. Select a suitable prediction model.
8. Make predictions for new patient data.

---

## 📂 Dataset

This project uses the **Heart Failure Clinical Records Dataset**.

### Dataset Information

| Information | Value |
|---|---:|
| 👥 Number of Patients | 299 |
| 📊 Number of Features | 12 |
| 🎯 Target Variable | `DEATH_EVENT` |
| 🧹 Missing Values | 0 |
| 🔁 Duplicate Rows | 0 |

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `anaemia` | Whether the patient has anaemia |
| `creatinine_phosphokinase` | CPK enzyme level in blood |
| `diabetes` | Whether the patient has diabetes |
| `ejection_fraction` | Percentage of blood leaving the heart during contraction |
| `high_blood_pressure` | Whether the patient has high blood pressure |
| `platelets` | Platelet count |
| `serum_creatinine` | Serum creatinine level |
| `serum_sodium` | Serum sodium level |
| `sex` | Sex of the patient |
| `smoking` | Whether the patient smokes |
| `time` | Follow-up period |
| `DEATH_EVENT` | Target variable |

### Target Variable

```text
0 → No death event
1 → Death event
