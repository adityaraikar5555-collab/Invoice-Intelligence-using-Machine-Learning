# 📊 Invoice Intelligence System using Machine Learning

> **An End-to-End Machine Learning Solution for Freight Cost Prediction and Intelligent Invoice Anomaly Detection using Python, SQLite, Scikit-learn, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

# 🚀 Project Overview

The **Invoice Intelligence System** is a Machine Learning-based analytics platform that automates freight cost estimation and identifies suspicious invoices from procurement records.

The project combines **Regression** and **Classification** techniques to assist organizations in logistics planning, invoice verification, procurement auditing, and financial analysis.

The system consists of two independent Machine Learning modules:

* 🚚 Freight Cost Prediction
* 🚩 Invoice Anomaly Detection (Invoice Flagging)

---

# 🛠 Technology Stack

* Python
* SQLite
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

# 🏗 System Architecture

```text
SQLite Database
        │
        ▼
 Data Preprocessing
        │
        ▼
 Feature Engineering
        │
        ▼
 Machine Learning Models
 ┌───────────────┴────────────────┐
 │                                │
 ▼                                ▼
Freight Cost Prediction      Invoice Flagging
(Regression)                 (Classification)
 │                                │
 └───────────────┬────────────────┘
                 ▼
      Streamlit Interactive Dashboard
```

---

# 📂 Project Structure

```text
Machine_Learning_Project/

│── Data_Project/
│      └── inventory.db

│── FreightCostPrediction/
│      ├── data_preprocessing.py
│      ├── modeling.py
│      ├── train.py
│      └── models/
│             └── predict_freight_model.pkl

│── invoice_flagging/
│      ├── data_processing.py
│      ├── modelling.py
│      ├── train.py
│      └── models/
│             ├── predict_flag_invoice.pkl
│             └── scaler.pkl

│── Notebooks/

│── app.py

│── requirements.txt

└── README.md
```

---

# 🚚 Freight Cost Prediction Module

## Objective

Estimate the expected freight cost of vendor invoices using Machine Learning.

The model learns the relationship between:

* Invoice Amount (Dollars)
* Freight Cost

to automatically predict future freight charges with high accuracy.

---

## Machine Learning Models Evaluated

* Linear Regression
* Decision Tree Regression
* Random Forest Regression

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📈 Model Comparison

| Model                    | MAE       | RMSE       | R² Score   |
| ------------------------ | --------- | ---------- | ---------- |
| **Linear Regression**    | **24.11** | **124.72** | **96.99%** |
| Decision Tree Regression | 32.97     | 150.31     | 95.63%     |
| Random Forest Regression | -         | 131.76     | 96.64%     |

---

## 🏆 Selected Model

**Linear Regression**

The trained model is automatically saved as:

```text
FreightCostPrediction/models/predict_freight_model.pkl
```

---

## Prediction Workflow

```text
Invoice Amount
       │
       ▼
Data Preprocessing
       │
       ▼
Linear Regression Model
       │
       ▼
Predicted Freight Cost
```

---

# 🚩 Invoice Flagging Module

## Objective

Automatically identify suspicious invoices using historical procurement and invoice information.

The system helps detect:

* Invoice mismatches
* Purchase inconsistencies
* Delayed receiving
* Payment irregularities
* Potential fraud

---

## Features Used

* Invoice Quantity
* Invoice Dollars
* Freight
* Days PO to Invoice
* Days to Pay
* Total Brands
* Total Item Quantity
* Total Item Dollars
* Average Receiving Delay

---

## Machine Learning Algorithm

Random Forest Classifier

Hyperparameter Optimization performed using:

* GridSearchCV
* 5-Fold Cross Validation

---

## 📊 Model Performance

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **96.30%** |
| F1 Score | **96.24%** |

Best Hyperparameters:

```text
criterion = gini
n_estimators = 100
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
```

Saved Model:

```text
invoice_flagging/models/predict_flag_invoice.pkl
```

---

# 📊 Streamlit Dashboard

The interactive Streamlit application allows users to:

* Predict Freight Cost
* Detect Suspicious Invoices
* Perform Real-time Machine Learning Predictions
* Analyze Invoice Data using a User-Friendly Interface

---
# 📸 Application Screenshots

## 🏠 Home Dashboard

The landing page provides an overview of the Invoice Intelligence System.

<p align="center">
<img src="https://raw.githubusercontent.com/adityaraikar5555-collab/Invoice-Intelligence-using-Machine-Learning/main/assests/HomeUI.png" width="95%">
</p>

---

## 🚚 Freight Cost Prediction

Users can instantly predict the expected freight cost.

<p align="center">
<img src="https://raw.githubusercontent.com/adityaraikar5555-collab/Invoice-Intelligence-using-Machine-Learning/main/assests/FreightCostPrediction.png" width="95%">
</p>

---

## ✅ Invoice Flagging (Normal Invoice)

The model classifies normal invoices.

<p align="center">
<img src="https://raw.githubusercontent.com/adityaraikar5555-collab/Invoice-Intelligence-using-Machine-Learning/main/assests/Invoice_flagging_normal.png" width="95%">
</p>

---

## 🚨 Invoice Flagging (Suspicious Invoice)

The model detects suspicious invoices.

<p align="center">
<img src="https://raw.githubusercontent.com/adityaraikar5555-collab/Invoice-Intelligence-using-Machine-Learning/main/assests/invoice_flagging_suspicious.png" width="95%">
</p>

---

# 💼 Business Applications

* Invoice Fraud Detection
* Financial Auditing
* Procurement Intelligence
* Logistics Cost Estimation
* Supply Chain Analytics
* Vendor Performance Analysis

---

# ▶️ Running the Project

Clone Repository

```bash
git clone <repository-url>
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Train Freight Prediction Model

```bash
python FreightCostPrediction/train.py
```

Train Invoice Flagging Model

```bash
python invoice_flagging/train.py
```

Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 🌟 Key Features

* End-to-End Machine Learning Pipeline
* SQLite Database Integration
* Freight Cost Prediction
* Invoice Anomaly Detection
* Automated Model Selection
* Hyperparameter Optimization
* Model Persistence using Joblib
* Interactive Streamlit Dashboard

---

# 🔮 Future Enhancements

* XGBoost and CatBoost Models
* Explainable AI (SHAP)
* REST API Integration
* Docker Containerization
* AWS/Azure Deployment
* Power BI Dashboard Integration
* Real-Time Invoice Monitoring

---

# 👨‍💻 Author

**Aditya M Raikar**

Computer Science Engineering

Machine Learning • Data Science • Python • SQL • Scikit-learn

---

### ⭐ If you found this project useful, consider giving it a **Star** on GitHub!
