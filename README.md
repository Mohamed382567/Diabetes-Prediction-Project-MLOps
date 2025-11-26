# Diabetes Prediction MLOps Deployment 🩺

This repository holds the **operational code** and assets required to deploy the Diabetes Prediction model (developed in [**https://github.com/Mohamed382567/Diabetes-Prediction-Project-using-Pima-Indains-Dataset**]) as a fully interactive web application using Streamlit.

**Goal:** To transform the theoretical model into a practical, live tool, demonstrating the crucial **Deployment** phase within the MLOps lifecycle, and ensuring that **all Feature Engineering and modeling steps are executed** on new user data.

### 🚀 Live Application & Model Details

The application applies the entire preprocessing pipeline (Imputation, Scaling, and all custom features) on user input to provide real-time risk assessment.

| Description | Link |
| :--- | :--- |
| **Live Streamlit App** | [https://diabetes-prediction-mohamed-elbaz-6ru5c5h8bb9bfbargwcdzt.streamlit.app/] |
| **Source Code (Modeling & Analysis)** | [https://github.com/Mohamed382567/Diabetes-Prediction-Project-using-Pima-Indains-Dataset] |

### 🛠️ Key Technical Components (Deployment Assets)

| File | Purpose / MLOps Role |
| :--- | :--- |
| **`app.py`** | **The Streamlit main entry point**. Contains the UI logic and the **in-app Feature Engineering pipeline** that strictly mirrors the training process. |
| **`requirements.txt`** | Lists the necessary Python libraries for Streamlit Cloud. |
| **`random_forest_model.joblib`** | The final trained **Random Forest** classifier. |
| **`iterative_imputer.joblib`** | The **Fitted MICE Imputer**—essential for handling missing (zero) values in user data exactly as the model was trained. |
| **`standard_scaler.joblib`** | The **Fitted StandardScaler**—ensures new inputs are normalized to the same distribution. |
| **`training_columns.joblib`** | **The column order file is crucial for MLOps,** ensuring that new user input data is ordered and passed to the model in the same order it was trained in (**preventing column mismatch errors**).

### ⚙️ MLOps Highlights & Engineering Challenges

| Highlight | Description |
| :--- | :--- |
| **Continuous Integration (CI) of Features** | Demonstrated by integrating **all Feature Engineering and modeling steps** directly into the website's logic (`app.py`), ensuring **strict consistency** between the training and serving environment. |
| **Operational Consistency** | The application ensures that all user inputs are processed using the saved **Fitted Tools** (Scaler and Imputer), preventing data or feature skew in production. |
| **Custom Decision Threshold** | The classification decision is made using a **40% threshold** (instead of the default 50%), highlighting awareness of the necessity to adapt models for critical, sensitive medical problems. |
| **Educational Note** | This project is primarily for educational purposes, demonstrating knowledge of **MLOps and Development (Deployment)**. The core idea is scalable if more resources and data become available. |

***
