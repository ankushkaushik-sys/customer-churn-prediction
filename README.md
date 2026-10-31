# Customer Churn Prediction & Retention Prioritization

A machine learning project that predicts the likelihood of a bank customer leaving and helps identify customers who may need retention attention. The project covers the complete workflow — from understanding the problem and preparing the data to model training, evaluation and a Streamlit web app.

## Project Objective

Customer churn affects customer retention and business performance. The objective is to use customer information to predict churn (`Exited = 1`) and provide an easy-to-use interface where a user can enter customer details and view the predicted churn probability.

## Project Workflow

| Step | Stage | What was done |
|---|---|---|
| 1 | Problem Identification | Defined customer churn prediction as a binary classification problem. |
| 2 | Data Understanding | Inspected the dataset, columns, data types, summary statistics and target distribution. |
| 3 | Data Cleaning | Removed duplicate records, checked missing values and handled invalid domain values. |
| 4 | Outlier Handling | Used the IQR method to identify extreme values and reviewed them before treatment. |
| 5 | Exploratory Data Analysis | Studied customer characteristics and churn patterns using plots and a correlation heatmap. |
| 6 | Feature Preparation | Excluded identifier columns and Gender from model inputs; encoded Geography and scaled numerical features. |
| 7 | Class Imbalance | Applied SMOTE to the training data to address the imbalance between churn and non-churn customers. |
| 8 | Model Selection | Compared classification models, including Logistic Regression, Decision Tree, Random Forest and Gradient Boosting. |
| 9 | Hyperparameter Tuning | Used `RandomizedSearchCV` with 3-fold cross-validation and ROC-AUC scoring to tune Gradient Boosting. |
| 10 | Model Evaluation | Evaluated the final model using classification metrics, a confusion matrix and an ROC curve. |
| 11 | Model Deployment | Saved the trained model, preprocessing object and feature-column order as `.pkl` files. |
| 12 | Streamlit UI | Built a simple web interface to collect customer details and display churn probability and risk category. |

## Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, Imbalanced-learn
- **Model tuning:** RandomizedSearchCV
- **Model persistence:** Joblib
- **UI:** Streamlit

## Dataset

The project uses a bank customer churn dataset with customer attributes such as:

`CreditScore`, `Geography`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

The target column is `Exited`:
- `0` — Customer did not exit
- `1` — Customer exited

Identifier fields such as `RowNumber`, `CustomerId` and `Surname` are not used as predictive features. `Gender` is also excluded from model inputs.

**Dataset limitation:** The available dataset is a static customer-level dataset and does not contain timestamps or customer history. Therefore, a chronological train-test split and true future-period churn validation could not be performed. A stratified random split was used instead.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   ├── Churn_Modelling_Messy_Practice.csv
│   └── cleaned_customer_churn.csv
│
├── notebooks/
│   ├── 01_DataCleaningPreprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_FS_ModelTrain.ipynb
│   ├── 04_ModelSelect_Train.ipynb
│   └── 05_HyperTuning.ipynb
│
├── app.py
├── final_churn_model.pkl
├── final_preprocessor.pkl
├── final_feature_columns.pkl
├── gradient_boosting_roc.png
└── README.md
```

## Run the Streamlit App

1. Clone the repository and open the project folder.
2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit joblib
```

3. Run the application:

```bash
streamlit run app.py
```

```bash
https://customer-churn-prediction-6fmyecf2ooqfktnjbvlpck.streamlit.app/
```

4. Enter the customer details and click **Predict Churn**.

The app displays the estimated churn probability and a risk label based on the model prediction.

## Model Files

The application loads these files from the project directory:

- `final_churn_model.pkl` — trained Gradient Boosting model
- `final_preprocessor.pkl` — fitted feature preprocessing object
- `final_feature_columns.pkl` — expected input feature order

Keep all three files in the same directory as `app.py`.

## Important Note

This is an educational project and a prototype for retention prioritization. The prediction is a model estimate, not a guarantee that a customer will leave. Any retention decision should involve human review. The dataset and validation approach also limit how confidently the model can be applied to real future customers.
