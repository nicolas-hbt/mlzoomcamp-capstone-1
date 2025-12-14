# 🎓 HR Analytics: Job Change Prediction (ML Zoomcamp Capstone)

## 📌 Problem Description

A company active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which the company conducts. 
Many people sign up for their training. 
Company wants to know which of these candidates are really looking to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates.

Information related to demographics, education, experience are in hands from candidates signup and enrollment.

**The Goal:**
The goal of this project is to build a classification model to predict the probability of a candidate looking for a new job or will work for the company, as well as interpreting affected factors on employee decision.

* **Target `0`:** Not looking for job change.
* **Target `1`:** Looking for a job change.

---

## 📊 Data Description

The dataset used for this project is the [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists) dataset from Kaggle.

It contains **19,158 samples** and **14 features** (including the target). The dataset includes a mix of numerical, categorical, and ordinal features.

### Features
* **`enrollee_id`**: Unique ID for candidate.
* **`city`**: City code.
* **`city_development_index`**: Scaled development index of the city (non-dimensional).
* **`gender`**: Gender of candidate.
* **`relevent_experience`**: Relevant experience in the field.
* **`enrolled_university`**: Type of University course enrolled if any.
* **`education_level`**: Education level of candidate.
* **`major_discipline`**: Education major discipline of candidate.
* **`experience`**: Candidate total experience in years.
* **`company_size`**: Number of employees in current employer's company.
* **`company_type`**: Type of current employer.
* **`last_new_job`**: Difference in years between previous job and current job.
* **`training_hours`**: Completed training hours.
* **`target`**: 0 – Not looking for job change, 1 – Looking for a job change.

---

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
* **Class Imbalance:** The target class is imbalanced, with only ~25% of candidates looking for a job change (Target=1).
* **Correlations:** `city_development_index` (CDI) showed the strongest negative correlation with the target (-0.34), indicating that candidates from cities with higher development indices are less likely to leave.
* **Distributions:** Analyzed distributions of training hours and experience levels.
* **Missing Values:** Identified significant missing data in `company_size`, `company_type`, and `gender`.

### 2. Preprocessing & Feature Engineering
All preprocessing steps were encapsulated in a `Scikit-Learn` pipeline to prevent data leakage and ensure reproducibility.

* **Cleaning:**
    * Converted ordinal strings (e.g., `experience` '>20' → 21, `last_new_job` 'never' → 0) into numerical values.
    * Fixed data types for IDs and Target variables.
* **Imputation:**
    * **Numerical:** Median imputation.
    * **Categorical:** Mode (most frequent) imputation.
* **Encoding:** Used `OneHotEncoder` for categorical variables (handling unknown categories).
* **Scaling:** Applied `StandardScaler` to numerical features.

### 3. Model Training & Tuning
The dataset was split into **80% train** and **20% validation** sets using **stratified sampling** to maintain the class ratio.

Four models were trained and tuned using `RandomizedSearchCV` (with 5-fold CV):
1.  **Logistic Regression** (Baseline)
2.  **Random Forest Classifier**
3.  **XGBoost Classifier**
4.  **Multi-Layer Perceptron (Neural Network)**

---

## 📈 Results & Evaluation

The primary evaluation metric selected was **ROC-AUC (Area Under the Curve)** because it is threshold-independent and robust for imbalanced classification problems.

| Model                | Validation ROC-AUC |
|:---------------------| :--- |
| **Random Forest**    | **0.782053** |
| **XGBoost**          | **0.781054** |
| MLP Classifier (DNN) | 0.772335 |
| Logistic Regression  | 0.770977 |

### 🏆 Final Model Selection: Random Forest

Random Forest was selected as the final model for deployment.
The final Random Forest model (tuned with `n_estimators`, `max_depth`, `min_samples_leaves`, and `min_samples_split`) achieved an **AUC of ~0.782** on the validation set.

---

# Setup & Installation

1. Clone the Repository:

```
git clone https://github.com/nicolas-hbt/mlzoomcamp-capstone-1.git
cd mlzoomcamp-capstone-1
```

2. Create a Virtual Environment and Install Dependencies:

```
# Install Pipenv if you haven't already
pip install pipenv

# Install project dependencies from Pipfile.lock
pipenv install
```

# Usage
There are three main ways to use this project:

1. Run the Prediction Server
This starts the Flask API server, which loads the pre-trained model to serve predictions.

```
python predict.py
```
The server will be running at http://0.0.0.0:9696 (or http://localhost:9696).

2. Test the Prediction Server
While the server is running (in a separate terminal), you can run the test script to send a sample request.

```
python predict_test.py
```
You should see a JSON response with predictions for 5 different people.
The test script sends the following JSON data to the /predict endpoint. 
You can use this structure for your own requests and change parameters as you wish:

```json
{
    "city": "city_21",
    "city_development_index": 0.624,
    "gender": "Male",
    "relevent_experience": "Has relevent experience",
    "enrolled_university": "no_enrollment",
    "education_level": "Masters",
    "major_discipline": "STEM",
    "experience": "5",
    "company_size": "50-99",
    "company_type": "Funded Startup",
    "last_new_job": "1",
    "training_hours": 45
}
```

3. (Optional) Re-train the Model
If you want to re-train the model from scratch:

```
python train.py
```

This will load aug_train.csv, run the preprocessing and cross-validation, and save a new model.bin file.

## 🐳 Alternative Usage (Docker)
You can also build and run this project as a Docker container using the provided Dockerfile.

Build the Docker Image From the root of the repository, run:

```
docker build -t job-change-predictor .
```

Run the Docker Container This command runs the container and maps your local port 9696 to the container's port 9696 .

```
docker run -d -p 9696:9696 job-change-predictor
```

The prediction server is now running and accessible at http://localhost:9696/predict. You can test it using predict_test.py as shown in the local usage section.
