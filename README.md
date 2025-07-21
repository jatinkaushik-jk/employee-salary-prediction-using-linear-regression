# Employee Salary Prediction using Machine Learning

## Overview
This project predicts whether an employee's salary is greater than 50K or less than or equal to 50K using various machine learning algorithms, with a focus on linear models and ensemble methods. The solution includes a Jupyter Notebook for data analysis and model training, as well as a user-friendly Streamlit web application for interactive predictions.

## Features
- Data preprocessing and cleaning (handling missing values, outlier removal, encoding)
- Model training and evaluation using multiple algorithms:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gradient Boosting
- Model comparison and selection of the best-performing model
- Interactive web app for single and batch predictions
- Batch CSV upload and downloadable results

## Dataset
- **File:** `final_data.csv`
- **Features Used:**
  - `age`: Age of the employee
  - `education`: Education level
  - `occupation`: Job role
  - `hours-per-week`: Number of working hours per week
  - `gender`: Gender of the employee
  - `salary`: Target variable (`<=50K` or `>50K`)

## Data Preprocessing
- Missing values in `occupation` replaced with 'Others'
- Outliers in `age` removed (kept ages 17–75)
- Label encoding for categorical features (`education`, `occupation`, `gender`)
- Feature scaling using `StandardScaler` (in pipelines)

## Model Training & Selection
- Data split into training and test sets (80/20)
- Multiple models trained and evaluated
- Best model (Gradient Boosting) selected based on accuracy
- Model saved as `best_model.pkl`

## Project Structure
```
Salary-Prediction-using-Machine-Learning/
├── employee_salary_prediction.ipynb  # Jupyter Notebook with full workflow
├── final_data.csv                   # Cleaned dataset
├── app.py                           # Streamlit web app (created by notebook)
├── best_model.pkl                   # Saved best model (created by notebook)
└── README.md                        # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Install Required Packages
Run the following command to install the necessary libraries:
```bash
pip install pandas scikit-learn matplotlib streamlit joblib
```

### 1. Running the Jupyter Notebook
1. Open `employee_salary_prediction.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells to:
   - Load and preprocess the data
   - Train and evaluate models
   - Save the best model as `best_model.pkl`
   - Generate the Streamlit app (`app.py`)

### 2. Using the Streamlit Web App
1. Ensure `final_data.csv`, `app.py`, and `best_model.pkl` are in the same directory.
2. Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
3. The app will open in your browser (default: http://localhost:8501).

#### App Features
- **Sidebar Inputs:** Enter employee details (age, education, occupation, hours per week, gender) to predict salary class.
- **Batch Prediction:** Upload a CSV file with the same columns for batch predictions. Download results as a CSV.

## User Guide
### Single Prediction
1. Launch the Streamlit app.
2. Use the sidebar to input employee details:
   - Age (17–75)
   - Education level (select from dropdown)
   - Occupation (select from dropdown)
   - Hours per week (1–99)
   - Gender (Male/Female)
3. Click **Predict Salary Class** to see the prediction (`<=50K` or `>50K`).

### Batch Prediction
1. Prepare a CSV file with columns: `age`, `education`, `occupation`, `hours-per-week`, `gender`.
2. Upload the file using the **Batch Prediction** section.
3. The app will display predictions and provide a download link for the results.

## Notes
- The app uses label encoding for categorical features. Ensure input values match those in the training data.
- If you encounter errors with batch prediction, check that your CSV uses valid values for `education`, `occupation`, and `gender`.

## License
This project is for educational purposes.

## Acknowledgements
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Matplotlib](https://matplotlib.org/) 