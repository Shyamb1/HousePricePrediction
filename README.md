# 🏠 House Price Prediction

An end-to-end Machine Learning project that predicts house prices using a tuned Random Forest Regression model.

## 🚀 Live Demo

[Open Live House Price Prediction App](https://housepriceprediction-ff3ec9kvpcg2wyddtaokxu.streamlit.app/)

## 📌 Project Overview

This project predicts residential house prices using the Kaggle House Prices dataset.

The complete machine learning workflow includes:

- Data cleaning
- Missing value handling
- Exploratory data analysis
- One-hot encoding
- Feature engineering
- Train-test splitting
- Linear Regression
- Random Forest Regression
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Feature importance analysis
- Model saving and loading
- Streamlit web application
- Cloud deployment

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Git
- GitHub

## 🤖 Machine Learning Models

### Linear Regression

| Metric | Score |
|---|---:|
| MAE | 20,383.82 |
| RMSE | 31,932.44 |
| R² Score | 0.8671 |

### Tuned Random Forest

| Metric | Score |
|---|---:|
| MAE | 17,778.25 |
| RMSE | 29,139.30 |
| R² Score | 0.8893 |

The Tuned Random Forest model performed better than Linear Regression on the test dataset.

## 📊 Important Features

The top features identified by the Random Forest model were:

1. OverallQual
2. GrLivArea
3. TotalBsmtSF
4. 2ndFlrSF
5. BsmtFinSF1
6. 1stFlrSF
7. LotArea
8. GarageArea
9. GarageCars
10. YearBuilt

`OverallQual` was the most important feature in the trained Random Forest model.

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Random Forest Regression
   ↓
Hyperparameter Tuning
   ↓
Cross Validation
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
Cloud Deployment
```

## 📂 Project Structure

```text
HousePricePrediction/
│
├── data/
│   ├── cleaned_train.csv
│   ├── data_description.txt
│   ├── processed_train.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── models/
│   ├── house_price_model.pkl
│   └── feature_columns.pkl
│
├── notes/
│   ├── day1.md
│   ├── day2.md
│   ├── day3.md
│   ├── day4.0.md
│   ├── day4.1.md
│   ├── day5.md
│   ├── day6.md
│   ├── day7.md
│   ├── day8.md
│   ├── day9.md
│   ├── day10.md
│   ├── day11.md
│   ├── day12.md
│   ├── day13.md
│   ├── day14.md
│   ├── day15.md
│   └── day16.md
│
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── visualization.py
│
├── .gitignore
├── app.py
├── main.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shyamb1/HousePricePrediction.git
```

### 2. Move into the project directory

```bash
cd HousePricePrediction
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Results

The tuned Random Forest model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| MAE | 17,778.25 |
| RMSE | 29,139.30 |
| R² Score | 0.8893 |

The model provides better performance than the Linear Regression baseline.

## 🌐 Deployment

The application has been deployed using Streamlit Community Cloud.

### Live Application

[Open Live House Price Prediction App](https://housepriceprediction-ff3ec9kvpcg2wyddtaokxu.streamlit.app/)

The deployed application allows users to enter house characteristics and receive an estimated house price.

## 💡 Key Findings

- Random Forest performed better than Linear Regression.
- `OverallQual` was the most influential feature.
- `GrLivArea` was the second most important feature.
- Basement area and living-area features significantly contributed to prediction.
- Garage-related features also influenced house prices.
- The tuned Random Forest achieved an R² score of approximately `0.8893`.

## 🎯 Future Improvements

Possible future improvements include:

- Trying Gradient Boosting models
- Testing XGBoost or LightGBM
- Advanced feature engineering
- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV
- Improving the Streamlit user interface
- Adding prediction confidence or uncertainty estimates
- Adding more visualizations
- Improving model performance

## 👨‍💻 Author

**Shyam Babu**

B.Tech CSE & AIML

---

⭐ If you find this project useful, consider giving the repository a star!