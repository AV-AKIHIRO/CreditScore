

# CreditScore Classification Project

## Overview
This project implements a machine learning system for credit score classification using various algorithms including Logistic Regression and Neural Networks. The goal is to predict credit scores (Good, Standard, Poor) based on customer financial and demographic data.

## Project Structure
```
CreditScore/
├── MLGroupProject_Part2.ipynb    # Main Jupyter notebook with complete analysis
├── ML_Group_Project_Report.pdf   # Detailed project report
└── README.md                     # This file
```

## Dataset Description
The project uses a credit score dataset with the following key features:

### Target Variable
- **Credit_Score**: Categorical variable with three classes
  - Good (0)
  - Poor (1) 
  - Standard (2)

### Key Features
- **Demographic**: Age, Profession, Name
- **Financial**: Income_Annual, Base_Salary_PerMonth, Monthly_Balance
- **Credit History**: Credit_Limit, Total_Credit_Cards, Total_Bank_Accounts
- **Payment Behavior**: Total_Delayed_Payments, Delay_from_due_date, Payment_Behaviour
- **Loan Information**: Total_Current_Loans, Loan_Type, Per_Month_EMI
- **Credit Metrics**: Credit_Mix, Ratio_Credit_Utilization, Credit_History_Age

## Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Removed underscores from numeric values
- **Missing Value Handling**: Filled missing values using median imputation
- **Outlier Removal**: Implemented IQR-based outlier detection and capping
- **Feature Engineering**: 
  - One-hot encoding for categorical variables
  - Standardization of numerical features

### 2. Models Implemented

#### Logistic Regression
- **Binary Classification**: Standard logistic regression
- **Multinomial Classification**: Using 'saga' solver
- **Performance**: ~62% accuracy on test set

#### Neural Networks
- **Architecture 1**: 48 → 24 → 3 neurons (ReLU activation)
- **Architecture 2**: 64 → 32 → 16 → 8 → 3 neurons (ReLU activation)
- **Performance**: ~68% accuracy on test set
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Categorical crossentropy

### 3. Model Performance
| Model | Test Accuracy | F1 Score |
|-------|---------------|----------|
| Logistic Regression | 62.39% | - |
| Neural Network (2 layers) | 66.72% | 0.661 |
| Neural Network (4 layers) | 68.31% | 0.686 |

## Key Findings
1. **Feature Importance**: Financial metrics like credit limit, monthly balance, and payment behavior are crucial predictors
2. **Data Quality**: Significant missing values in several features required careful handling
3. **Model Performance**: Neural networks outperformed logistic regression, suggesting non-linear relationships in the data
4. **Class Distribution**: The dataset shows imbalanced classes, which may affect model performance

## Usage Guidelines

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Running the Analysis
1. Open `MLGroupProject_Part2.ipynb` in Jupyter Notebook
2. Ensure the dataset files are in the correct path:
   - `train (3).csv` for training data
   - `test (4).csv` for test data
3. Run all cells sequentially to reproduce the analysis

### Data Requirements
- Training data: 80,000 samples with 27 features
- Test data: 20,000 samples with 26 features (excluding target)
- All numerical features should be properly scaled
- Categorical features should be encoded

### Model Deployment Considerations
- **Feature Scaling**: Use the same StandardScaler fitted on training data
- **Categorical Encoding**: Apply the same encoding scheme used during training
- **Missing Values**: Implement the same imputation strategy
- **Model Persistence**: Save trained models using pickle or joblib

## Future Improvements
1. **Feature Selection**: Implement techniques to identify most important features
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimization
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
5. **Class Imbalance**: Address using techniques like SMOTE or class weights

## Contributors
This is a group project for machine learning classification. See the detailed report in `ML_Group_Project_Report.pdf` for complete methodology and results.
