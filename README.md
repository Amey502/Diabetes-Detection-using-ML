# Diabetes Detection using Machine Learning

This project uses machine learning to predict the likelihood of a patient having diabetes based on clinical data such as age, gender, BMI, blood sugar levels, hypertension, and more. Logistic Regression, a popular classification algorithm, is used to build and evaluate the model.

## Dataset

- **Source**: [100000 Diabetes Clinical Dataset - Kaggle](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)
- The dataset contains over 100,000 patient records with features like:
  - `age`
  - `gender`
  - `race`
  - `bmi`
  - `blood sugar level`
  - `hypertension`
  - `diabetes` (target label: 0 = No, 1 = Yes)

## Objective

To develop a supervised machine learning model using Logistic Regression that can predict whether a patient has a high likelihood of being diabetic based on the clinical features provided.

## Technologies Used

- **Python**
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-learn** for:
  - Logistic Regression
  - Data preprocessing (e.g., label encoding, train-test split)
  - Model evaluation metrics (accuracy, confusion matrix, ROC curve)

## Model Training Steps

1. Load and explore the dataset.
2. Handle missing values and encode categorical variables.
3. Split the dataset into training and testing sets.
4. Train a logistic regression model.
5. Evaluate the model using:
   - Accuracy
   - Confusion Matrix
   - ROC-AUC Score

## Sample Results
- Accuracy: 85-95% (multiple tweaked models used)
- ROC-AUC Score: ~0.95
- The model highlights the influence of features like BMI, blood sugar, and hypertension in predicting diabetes risk.

## Future Improvements
- Experiment with other models like Random Forest or XGBoost.
- Use feature selection techniques to improve performance.
- Build a web-based interface for easier usage by non-technical users.
