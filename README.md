# First Payment Default Prediction

This repository contains a machine learning project aimed at predicting **First Payment Default (FPD)** using an anonymized dataset with financial, employment, behavioral, and location-based features. The goal is to develop a robust, scalable model that generalizes well to unseen data while meeting key performance metrics.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Project Structure](#project-structure)
5. [Usage Instructions](#usage-instructions)
6. [Results](#results)
7. [Contributions](#contributions)

---

## **Overview**

The **First Payment Default (FPD)** prediction problem is critical in the financial industry, where accurately predicting whether a customer will default on their first payment can greatly improve risk management and decision-making processes. This project focuses on building a machine learning model to predict FPD using anonymized data that includes:

- **Transaction Data**: Historical financial transactions and payment behavior.
- **Employment and Income History**: Job status, salary trends, and income-related information.
- **Behavioral Metrics**: Online activity, social media presence, and user behavior analysis.
- **Business and Location Characteristics**: Business type, geographic influences, and other contextual data.
- **Identity Verification and Communication Activity**: Verification steps, communication channels, and interaction history.

---

## **Dataset Description**

### **Files**
- **FDP.csv**: The full dataset containing all available features. Participants must split this into training and test sets as needed.

### **Columns**
The dataset consists of **87 anonymized columns**, including:
- **hashed_id**: Unique identifier for each entity (derived from CURP using a secure hashing method). This column should not be used for prediction.
- **feature_1 to feature_86**: Anonymized features derived from multiple domains.
- **target**: Binary target variable (`1` = first payment default, `0` = no default).

### **Data Sources**
The dataset is fully anonymized and includes features derived from:
- Transaction Data
- Employment and Income History
- Financial Behavior
- Business and Location Characteristics
- Digital Presence and Behavioral Metrics
- Identity Verification and Communication Activity

---

## **Evaluation Metrics**

The model's performance is evaluated based on the following metrics:
1. **Kolmogorov-Smirnov (KS) Score**: ≥ 0.3
2. **ROC AUC Score**: ≥ 0.7
3. **Accuracy**: ≥ 0.9

---

## **Project Structure**

The repository is organized as follows:
- `/data`: Contains the dataset (`FDP.csv`) and a description of the data.
- `/notebooks`: Jupyter notebooks for exploratory data analysis, feature engineering, and model training.
- `/models`: Trained models saved in `.pkl` or `.h5` format.
- `/utils`: Utility scripts for preprocessing and custom metrics.
- `report.ipynb`: Final results, visualizations, and conclusions.

---
