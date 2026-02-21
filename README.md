## End to End machine learing project

# Student Exam Performance Predictor

## Project Overview

The Student Exam Performance Predictor is a supervised machine learning project designed to predict students’ exam scores based on academic and demographic attributes.

This project demonstrates an end-to-end machine learning pipeline including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

---

## Problem Statement

The objective of this project is to build a predictive model that estimates student exam performance using relevant features. The model helps identify patterns and factors influencing academic outcomes.

---

## Dataset Description

The dataset includes the following features:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Math, Reading, and Writing Scores  

The target variable is the selected subject score.

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn
- Flask

---

## Methodology

1. Data Cleaning and Preprocessing  
2. Encoding Categorical Variables  
3. Feature Scaling  
4. Train-Test Split  
5. Model Training (Linear Regression / Random Forest)  
6. Performance Evaluation using MAE, MSE, and R² Score  

---

## Results

The model was evaluated using standard regression metrics:

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- R² Score  

The trained model demonstrates reliable predictive performance on unseen test data.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/amey-gif/Student-Exam-Performance-Predictor.git
cd Student-Exam-Performance-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Future Enhancements

- Hyperparameter tuning  
- Model deployment using Flask or Streamlit  
- Integration with a web-based dashboard  

---

## Author

Amey Parab  
GitHub: https://github.com/amey-gif
