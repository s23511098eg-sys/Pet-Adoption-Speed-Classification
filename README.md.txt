# 🐾 Pet Adoption Speed Classification

## 📌 Project Overview

This project aims to predict how quickly a pet will be adopted based on its profile, health condition, and listing information.

The project uses Machine Learning classification algorithms to predict one of five adoption speed classes (0–4), using data from PetFinder.my.

## 🎯 Project Objectives

- Explore and analyze the pet adoption dataset.
- Clean and preprocess the data.
- Perform feature engineering.
- Train and compare different Machine Learning classification models.
- Evaluate model performance using multiple metrics.
- Build an interactive web application using Streamlit.

## 📊 Dataset

The dataset contains approximately 15,000 pet listings and 24 columns.

It includes information about:

- Pet profile: Type, Age, Breed, Gender, and Color.
- Health: Vaccinated, Dewormed, Sterilized, and Health.
- Listing information: Quantity, Fee, State, Photo Amount, and Video Amount.
- Target variable: AdoptionSpeed.

### Target Classes

| Class | Adoption Speed |
|---|---|
| 0 | Same day |
| 1 | First week |
| 2 | First month |
| 3 | 2nd–3rd month |
| 4 | Not adopted |

## 🧹 Data Cleaning & Feature Engineering

The following steps were performed:

- Handled missing values in Name and Description.
- Created `HasName` to indicate whether a pet has a name.
- Created `DescLength` to calculate the description length.
- Created `RescuerPosts` using frequency encoding.
- Removed `PetID` because it is a unique identifier.
- Removed unnecessary text columns after feature extraction.

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and discover patterns, including:

- Adoption speed distribution.
- Dogs vs. cats analysis.
- Age distribution.
- Health and pet characteristics.
- Correlation heatmap.

## ⚙️ Data Preprocessing

The preprocessing steps included:

- Separating features (X) and target (y).
- Applying One-Hot Encoding to categorical features.
- Keeping Breed1 and Breed2 as numerical features.
- Splitting the data into 80% training and 20% testing.
- Using stratified splitting with `random_state=42`.
- Applying StandardScaler, fitted on training data only.

## 🤖 Machine Learning Models

Three classification models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier (300 estimators)

## 📈 Model Evaluation & Results

The models were evaluated using Training Accuracy, Testing Accuracy, and Macro F1-score.

| Model | Train Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 39.5% | 37.8% | 0.286 |
| Decision Tree | 99.8% | 35.0% | 0.307 |
| Random Forest | 99.8% | 43.3% | 0.366 |

### Best Model: Random Forest

Random Forest achieved the best test accuracy (43.3%) and Macro F1-score (0.366) among the three models.

## 💡 Feature Importance

The most important features identified by the Random Forest model include:

1. DescLength
2. RescuerPosts
3. Age
4. PhotoAmt
5. Breed1

The results suggest that listing quality, including detailed descriptions and photos, is important for understanding adoption speed.

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application with two main pages:

### 1. Data Exploration
- Dataset overview.
- Missing values analysis.
- Adoption speed distribution.
- Dogs vs. cats charts.
- Age distribution.
- Correlation heatmap.

### 2. Prediction
- Enter a pet's profile and characteristics.
- Predict the expected adoption speed.
- Display prediction probabilities for all five classes.

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 📁 Project Structure

```text
Pet-Adoption-Speed-Classification/
│
├── app.py
├── train_model.py
├── Pet_Adoption_Classification.ipynb
│
├── model.pkl
├── scaler.pkl
├── feature_columns.json
│
├── train.csv
├── breed_labels.csv
├── color_labels.csv
├── state_labels.csv
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── presentation/
    └── Pet_Adoption_Presentation.pptx
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Pet-Adoption-Speed-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

## 👩‍💻 Project Purpose

This project demonstrates the application of Machine Learning classification techniques to a real-world pet adoption problem, from data exploration and preprocessing to model evaluation and interactive deployment.س