# 🏠 House Price Prediction using XGBoost

## 📌 Project Overview
This project focuses on predicting house prices using machine learning techniques.  
The **California Housing Dataset** is used to build a regression model that estimates median house prices based on various socio-economic and geographical features.

The model is implemented using **XGBoost Regressor**, which provides high accuracy and strong generalization performance.

---

## 🎯 Objectives
- Load and explore a real-world housing dataset
- Perform data analysis and visualization
- Understand correlations between features
- Build and train a regression model
- Evaluate model performance using standard metrics

---

## 📊 Dataset Description
- **Dataset Name:** California Housing Dataset  
- **Source:** OpenML  
- **Number of Features:** 8  
- **Target Variable:** Median House Value (`Price`)

### Features:
| Feature | Description |
|-------|------------|
| MedInc | Median income in block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block group population |
| AveOccup | Average house occupancy |
| Latitude | Latitude coordinate |
| Longitude | Longitude coordinate |

---

## 🛠️ Technologies Used
- **Python 3**
- **Pandas & NumPy** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Data splitting & evaluation
- **XGBoost** – Machine learning model

---

## ⚙️ Project Workflow
1. Load dataset using OpenML
2. Convert data into Pandas DataFrame
3. Perform exploratory data analysis (EDA)
4. Analyze correlations using heatmap
5. Split dataset into training and testing sets
6. Train XGBoost regression model
7. Evaluate model using R² Score and MAE
8. Visualize actual vs predicted prices

---

## 📈 Model Performance

| Metric | Training | Testing |
|------|---------|---------|
| R² Score | 0.93 | 0.85 |
| Mean Absolute Error (MAE) | ~21,000 | ~30,000 |

✔ The model shows **strong predictive performance**  
✔ Minimal overfitting  
✔ Good generalization on unseen data

---

## 📊 Visualization
- Correlation heatmap to analyze feature relationships
- Scatter plot of actual vs predicted house prices

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

2️⃣ Run the Python Script / Notebook
python house_price_prediction.py

🧠 Key Learnings

Handling deprecated datasets in scikit-learn

Using OpenML for reliable data access

Importance of choosing the right regression model

Understanding underfitting vs overfitting

Evaluating models using R² and MAE

📌 Future Enhancements

Hyperparameter tuning using GridSearchCV

Cross-validation for robustness

Feature importance visualization

Deploying the model using Flask or Streamlit
