# 🏎️ F1 Race Winner Prediction Model

## 📌 Project Overview
This project is a Machine Learning pipeline built in Python that predicts the winner of a Formula 1 Grand Prix. By analyzing historical race data, driver information, and starting grid positions, the model identifies patterns to forecast race outcomes. It serves as a practical introduction to data engineering, data preprocessing, and predictive AI modeling.

## ✨ Features
* **Automated Data Gathering:** Fetches official historical race results and telemetry using the `fastf1` API.
* **Data Preprocessing:** Cleans and structures raw race data, utilizing `scikit-learn`'s Label Encoding to translate text-based categorical data (driver names, constructor teams) into machine-readable numerical features.
* **Predictive AI Brain:** Utilizes a Random Forest Classifier trained on an 80/20 data split to calculate the probability of a driver winning based on their starting conditions.
* **Custom Scenarios:** Allows users to input hypothetical race scenarios (e.g., "What if Lando Norris starts 2nd for McLaren?") to receive instant win/loss predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`
* **Machine Learning:** `scikit-learn` (RandomForestClassifier, LabelEncoder)
* **APIs & Data Sourcing:** `fastf1`
* **Environment:** Designed to run seamlessly in Google Colab or any Jupyter Notebook environment.

## 🚀 Getting Started

### 1. Set Up Your Environment
The easiest way to run this project is through Google Colab, which requires zero local installation. 

### 2. Install Dependencies
If running locally or in a new Colab notebook, install the required libraries:
```bash
pip install fastf1 scikit-learn pandas
