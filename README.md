# Telecom Customer Churn Prediction

Machine Learning project to predict customer churn in the telecom industry using multiple classification models — now deployed as an interactive web app.

## 🔗 Live Demo

**Try it here:** [https://telecom-churn-prediction-vyoj2n9xtccfvu4uhyluzf.streamlit.app/](https://telecom-churn-prediction-vyoj2n9xtccfvu4uhyluzf.streamlit.app/)

## Overview

This project analyzes telecom customer data to predict which customers are likely to churn (leave the service), helping businesses take proactive retention measures.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## What I Did

- Performed complete data preprocessing (missing values, label encoding, one-hot encoding, feature scaling)
- Conducted Exploratory Data Analysis (EDA)
- Built and compared multiple models: Logistic Regression, Random Forest, XGBoost
- Created visualizations: churn distribution, correlation heatmaps, ROC curves, confusion matrices, model comparison charts
- Deployed the best-performing model as an interactive Streamlit web app for real-time churn prediction

## Results

- **Best performing model:** Logistic Regression
- **Accuracy:** 80.7%
- **ROC-AUC Score:** 84.16%

## Project Structure

```
├── CUSTOMER_CHURN_PREDICTION.ipynb   # Full analysis, EDA, and model training
├── app.py                            # Streamlit web app for live predictions
├── churn_model.pkl                   # Trained model + preprocessing pipeline
├── requirements.txt                  # Python dependencies
└── README.md
```

## How to Run

### Notebook (training / analysis)
1. Clone this repo
2. Open `CUSTOMER_CHURN_PREDICTION.ipynb` in Jupyter or Google Colab
3. Install required libraries: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
4. Run all cells sequentially

### Web App (deployment)
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Open the local URL shown in your terminal, fill in customer details, and get an instant churn prediction

## Contact

Javeria Akmal — [@Javeria-Akmal](https://github.com/Javeria-Akmal) | LinkedIn
