# CreditScore Classification Project

## Overview

This project implements and evaluates machine learning models for credit score classification. It preprocesses customer financial and demographic data and compares the performance of Logistic Regression against several Neural Network architectures to predict credit scores (Good, Standard, Poor).

## Project Structure

```
CreditScore/
├── MLGroupProject_Part2.ipynb    # Main Jupyter notebook with complete analysis
├── ML_Group_Project_Report.pdf   # Detailed project report
└── README.md                     # This file
```

## Dataset Description

The project uses a credit score dataset with 80,000 training samples and 20,000 test samples.

### Target Variable

- **Credit_Score**: Categorical variable, label-encoded as:
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

- **Data Cleaning**: Removed underscores (`_`) and other artifacts (`!@9#%8`, `_______`) from string and numeric-like columns.
- **Missing Value Handling**: Filled missing numerical features (`Base_Salary_PerMonth`, `Monthly_Balance`, etc.) using **median imputation**.
- **Outlier Removal**: Implemented capping for numerical features like `Age`, `Income_Annual`, and `Credit_Limit` based on the Interquartile Range (IQR).
- **Feature Engineering**:
  - Converted `Credit_History_Age` (e.g., "24 Years and 1 Months") into total months.
  - **One-hot encoding**: Applied to `Payment_of_Min_Amount` and `Profession`.
  - **Custom multi-hot encoding**: Applied to `Loan_Type` to handle entries with multiple loan types.
  - **Label encoding**: Applied to `Month`, `Credit_Mix`, and `Payment_Behaviour`.
- **Feature Scaling**: Standardized all numerical features using `StandardScaler`.

### 2. Models Implemented

#### Logistic Regression

- Several variations were tested, including standard, multinomial (`saga` solver), L1, L2, and weighted.
- All variations achieved similar performance.
- **Performance**: ~62.4% accuracy on the test set.

#### Neural Networks

- **Architecture 1 (Sigmoid)**:
  - **Layers**: 48 → 24 → 3 (Sigmoid activation)
  - **Dropout**: 0.2 applied after the first two layers.
- **Architecture 2 (ReLU)**:
  - **Layers**: 48 → 24 → 3 (ReLU activation)
- **Architecture 3 (Deeper ReLU)**:
  - **Layers**: 64 → 32 → 16 → 8 → 3 (ReLU activation)
- **Optimizer**: **Adam** (learning rate 0.001) for all networks.
- **Loss Function**: **Categorical Crossentropy** (since this is a multi-class problem).

### 3. Model Performance

| Model | Test Accuracy | F1 Score (Weighted) | Notebook Cell(s) |
|-------|---------------|---------------------|------------------|
| Logistic Regression | 62.39% | (not calculated) | 29-38 |
| Neural Network (2-layer, Sigmoid) | 66.72% | 0.661 | 35-38 |
| Neural Network (2-layer, ReLU) | 68.24% | 0.683 | 40-42 |
| **Neural Network (4-layer, ReLU)** | **68.31%** | **0.686** | **83-82** |

## Key Findings

1. **Data Quality**: The dataset contained significant missing values (e.g., in `Base_Salary_PerMonth`, `Monthly_Balance`, `Loan_Type`) and required extensive cleaning of non-standard string values.
2. **Model Performance**: All Neural Network architectures significantly outperformed Logistic Regression, indicating complex, non-linear relationships in the data.
3. **Activation Function**: For the 2-layer network, **ReLU** activation (68.24% accuracy) provided a noticeable improvement over **Sigmoid** activation (66.72% accuracy).
4. **Model Depth**: A deeper 4-layer network (68.31%) performed slightly better than a 2-layer network (68.24%), suggesting a marginal benefit from added complexity.

## Usage Guidelines

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras mord
```

### Running the Analysis

1. Open `MLGroupProject_Part2.ipynb` in Jupyter Notebook.
2. Ensure the dataset files are in the correct path:
   - `train (3).csv` for training data
   - `test (4).csv` for test data
3. Run all cells sequentially to reproduce the analysis.

### Data Requirements

- Training data: 80,000 samples
- Test data: 20,000 samples
- All numerical features must be scaled, and categorical features encoded as done in the notebook.

### Model Deployment Considerations

- **Feature Scaling**: Use the same StandardScaler fitted on training data
- **Categorical Encoding**: Apply the same encoding scheme used during training
- **Missing Values**: Implement the same imputation strategy
- **Model Persistence**: Save trained models using pickle or joblib

## Future Improvements

1. **Feature Importance**: Implement feature selection (e.g., using SHAP or permutation importance) to identify and remove non-predictive features.
2. **Class Imbalance**: Investigate target distribution by plotting class counts and, if needed, address using SMOTE or class weights in the neural network.
3. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize neural network parameters (e.g., layer size, learning rate, dropout).
4. **Ensemble Methods**: Experiment with tree-based ensemble models like Random Forest or XGBoost, which are often strong performers on tabular data.
5. **Cross-Validation**: Implement k-fold cross-validation for the neural network models to get a more robust estimate of their performance.

## Contributors

This is a group project for machine learning classification. See the detailed report in `ML_Group_Project_Report.pdf` for complete methodology and results.
