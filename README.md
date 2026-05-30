# Vendor Invoice Intelligence System
Freight Cost Prediction & Invoice Risk Flagging


📌 Table of Contents
* [Project Overview](#project-overview)
* [Business Objectives](#business-objectives)
* [Data Sources](#data-sources)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Models Used](#models-used)
* [Evaluation Metrics](#evaluation-metrics)
* [Application](#application)
* [Project Structure](#project-structure)
* [How to Run This Project](#how-to-run-this-project)
* [Author & Contact](#author-&-contact)

---

📌 Project Overview
This project implements an end-to-end machine learning system designed to support finance teams by:
1. Predicting expected freight cost for vendor invoices.
2. Flagging high-risk invoices that require manual review due to abnormal cost, freight, or operational patterns.

---

🎯 Business Objectives

### 1. Freight Cost Prediction (Regression)
**Objective:**
Predict the expected freight cost for a vendor invoice using quantity, invoice value, and historical behavior.

**Why it matters:**
* Freight is a non-trivial component of landed cost.
* Poor freight estimation impacts margin analysis and budgeting.
* Early prediction improves procurement planning and vendor negotiation.

### 2. Invoice Risk Flagging (Classification)
**Objective:**
Predict whether a vendor invoice should be flagged for manual approval due to abnormal cost, freight, or delivery patterns.

**Why it matters:**
* Manual invoice review does not scale.
* Financial leakage often occurs in large or complex invoices.
* Early risk detection improves audit efficiency and operational control.

---

📁 Data Sources
Data is stored in a relational SQLite database (`inventory.db`) with the following tables:
* `vendor_invoice` – Invoice-level financial and timing data
* `purchases` – Item-level purchase details
* `purchase_prices` – Reference purchase prices
* `begin_inventory`, `end_inventory` – Inventory snapshots

SQL aggregation is used to generate invoice-level features.

---

# Tools and Technologies
* **Programming Language:** Python 3.14
* **Database Management:** SQLite3 / SQL (Advanced Joins & Aggregations)
* **ML Frameworks:** Scikit-Learn, Joblib
* **Data Science Stack:** Pandas, NumPy, SciPy (Statistical Testing)
* **Visualization Stack:** Matplotlib, Seaborn
* **UI/Deployment Layer:** Streamlit (Web Analytics Portal)

---

# Methods
1. **Data Aggregation:** Leveraged complex SQL Joins, Group By expressions, and the `JulianDay()` function to convert raw string dates into operational duration metrics (e.g., Average Receiving Delay, Days to Pay).
2. **Hypothesis Testing (T-Test):** Applied `scipy.stats.ttest_ind` to calculate statistical significance between flagged and normal invoice distributions, dropping non-significant features to optimize model performance.
3. **Feature Scaling:** Mitigated huge value variances across disparate business metrics by applying a robust `StandardScaler`.
4. **Hyperparameter Tuning:** Implemented `GridSearchCV` optimized against an `F1-Score` metric to robustly handle class imbalance and systematically minimize False Positives.

---



📊 Exploratory Data Analysis (EDA)
EDA focuses on business-driven questions, such as:
* Do flagged invoices have higher financial exposure?
* Does freight scale linearly with quantity?
* Does freight cost depend on quantity?

Statistical tests (t-tests) are used to confirm that flagged invoices differ meaningfully from normal invoices.

---

🤖 Models Used

### Regression (Freight Prediction)
* Linear Regression (baseline)
* Decision Tree Regressor
* Random Forest Regressor (final model)

### Classification (Invoice Flagging)
* Logistic Regression (baseline)
* Decision Tree Classifier
* Random Forest Classifier (final model with GridSearchCV)

Hyperparameter tuning is performed using GridSearchCV with F1-score to handle class imbalance.

---

📈 Evaluation Metrics

### Freight Prediction
* MAE
* RMSE
* R² Score

### Invoice Flagging
* Accuracy
* Precision, Recall, F1-score
* Classification report
* Feature importance analysis

---

# Key Insights
* **Bulk Ordering Economies of Scale:** Statistical **T-Test** and quantile evaluation revealed that high-volume orders yield lower freight rates. Bulk (High Quantity) orders average a freight cost of only $0.04 per unit, compared to $0.09 per unit for low-quantity orders.
* **Feature Collinearity:** Exploratory Data Analysis (EDA) established a powerful 96% linear correlation between `Dollars` and `Freight` variables, highlighting Linear Regression as the most stable baseline model for cost projection.

🖥️ End-to-End Application
A Streamlit application demonstrates the complete pipeline:
* Input invoice details
* Predict expected freight
* Flag invoices in real time
* Provide human-readable explanations

---

# Results & Conclusion
* Successfully transitioned flat notebook code into clean, modular production scripts (`data_preprocessing.py`, `modelling_evaluation.py`, `train.py`).
* For the regression problem, simple **Linear Regression** outperformed complex ensemble structures by generating the lowest MAE and a high $R^2$ score (~97%).
* For the classification task, hyperparameter tuning via **GridSearchCV** minimized costly False Positives, pushing model accuracy to **89%** and ensuring reliable corporate automated auditing.
🗂️ Project Structure
```text

inventory-invoice-analytics/
├── data/
│   └── inventory.db
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── scaler.pkl
│   └── predict_flag_invoice.pkl
├── notebooks/
│   ├── Invoice Flagging.ipynb
│   └── Predict Freight Cost.ipynb
├── app.py
├── README.md
└── .gitignore




# Author & Contact
* **Name:** Prashant Gautam
* **Email:** prashantgautam962003@gmail.com
* **LinkedIn:** https://www.linkedin.com/in/prashant-gautam-22b123281/


