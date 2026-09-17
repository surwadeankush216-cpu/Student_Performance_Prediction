# Student Performance Prediction Using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's Performance Index using academic and lifestyle-related factors.

The project compares multiple Machine Learning regression models and selects the best-performing model based on evaluation metrics.

## 🎯 Objective

To develop a Machine Learning system that can predict student performance based on:

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

## 📊 Dataset

- Dataset: Student Performance Dataset
- Records: 10,000
- Input Features: 5
- Target Variable: Performance Index
- Missing Values: None
- Training Data: 80%
- Testing Data: 20%

## 🤖 Machine Learning Models

The following models were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)

## 📈 Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.6111 | 2.0206 | 0.9890 |
| Decision Tree | 2.3378 | 2.9686 | 0.9762 |
| Random Forest | 1.8147 | 2.2713 | 0.9861 |
| SVR | 1.8300 | 2.3210 | 0.9855 |

Linear Regression achieved the highest R² score among the tested models.

## 🖥️ Application

A Streamlit-based web application was developed to allow users to enter student details and obtain a predicted Performance Index.

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt