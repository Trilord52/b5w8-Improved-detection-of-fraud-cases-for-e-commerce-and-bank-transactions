# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Project Status
- **Task 1:** Data cleaning, EDA, feature engineering, geolocation, and class imbalance handling ✅
- **Task 2:** Model building and evaluation ⏳
- **Task 3:** Model explainability and reporting ⏳

## Project Overview
This project delivers a comprehensive, business-driven solution for detecting fraudulent transactions in e-commerce and banking.This project builds robust machine learning models to detect fraudulent transactions in e-commerce and banking. The workflow 
includes data cleaning, exploratory data analysis (EDA), feature engineering, geolocation analysis, handling class imbalance, 
and saving processed data for modeling. All steps are documented for clarity, business relevance, and alignment with best 
practices in fraud detection. The pipeline is designed for clarity, reproducibility, and actionable insights, leveraging:
- **Pandas, NumPy, Matplotlib, Seaborn** for data analysis and visualization
- **scikit-learn, imbalanced-learn** for feature engineering, modeling, and class imbalance handling
- **Jupyter/Colab** for interactive, well-documented analysis

### Business Need & Goals
- Accurately identify fraudulent transactions to minimize financial loss and improve customer trust
- Balance security with user experience by reducing false positives and negatives
- Provide explainable, business-relevant insights for stakeholders

## Table of Contents
- [Project Structure](#project-structure)
- [Setup & Reproducibility](#setup--reproducibility)
- [Workflow & Deliverables](#workflow--deliverables)
- [Feature Engineering Rationale](#feature-engineering-rationale)
- [Model Selection Strategy](#model-selection-strategy)
- [Handling Class Imbalance](#handling-class-imbalance)
- [EDA & Feature Engineering Insights](#eda--feature-engineering-insights)
- [Class Imbalance Strategy](#class-imbalance-strategy)
- [Processed Data](#processed-data)
- [Running in Google Colab](#running-in-google-colab)
- [Business Context & Rubric Alignment](#business-context--rubric-alignment)
- [Documentation and Collaboration](#documentation-and-collaboration)
- [References](#references)
- [Contributing, Author, License](#contributing-author-license)

## Project Structure
```
├── data/
│   ├── raw/         # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/   # Cleaned and feature-engineered data (ready for modeling)
├── notebooks/       # Jupyter/Colab notebooks for EDA and analysis
├── src/             # Source code (scripts, functions)
├── reports/         # Project reports and deliverables
├── figures/         # Plots and visualizations
├── utils/           # Utility scripts
├── requirements.txt # Python dependencies
├── .gitignore       # Excludes data, venv, and sensitive files
└── README.md        # Project documentation
```

## Setup & Reproducibility
1. **Clone the repository**
2. **Create and activate a virtual environment** (optional but recommended)
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Place datasets in `data/raw/`** (already included if you cloned the full repo)
5. **Run the notebook** `notebooks/01_task1_data_cleaning_eda.ipynb` for Task 1 analysis
   - For Google Colab, upload the notebook and mount your Google Drive as needed

## Workflow & Deliverables
### Task 1: Data Preparation & EDA
- **Data Cleaning & Preprocessing:**
  - Handle missing values, duplicates, and data type corrections
  - Document all steps for transparency
- **Exploratory Data Analysis (EDA):**
  - Univariate and bivariate analysis
  - Visualizations of fraud patterns (class distribution, purchase value, age, source, browser, country)
  - Business-relevant interpretations
- **Feature Engineering:**
  - Create features such as hour_of_day, day_of_week, time_since_signup, transaction_count, transactions_last_24h
  - Each feature is justified with a business hypothesis
- **Geolocation Analysis:**
  - Map IP addresses to countries and analyze country-level fraud risk
- **Class Imbalance Handling:**
  - Identify and address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
  - Justify the approach and document results
- **Processed Data:**
  - Save all cleaned and engineered datasets in `data/processed/` for modeling
- **Documentation:**
  - All steps are explained in markdown cells, with clear summaries and business context

## Feature Engineering Rationale
Each feature was engineered based on domain knowledge, EDA findings, and the goal of improving the model's ability to detect subtle fraud patterns:

- **hour_of_day:** Captures the hour of transaction; fraud may occur at unusual times (e.g., late at night).
- **day_of_week:** Certain days may see more fraud due to business cycles or attacker behavior.
- **time_since_signup:** Measures how quickly a user transacts after signup; fraudsters often act quickly to avoid detection.
- **transaction_count:** Total transactions per user; helps distinguish between normal and abnormal user behavior.
- **transactions_last_24h:** Captures transaction velocity; sudden spikes can indicate fraud.

Each feature is not only technically relevant but also directly tied to business hypotheses about how fraudsters behave differently from legitimate users.

## Model Selection Strategy
For the modeling phase, both interpretability and predictive power are prioritized:

- **Logistic Regression:** Used as a baseline for its interpretability and ability to provide clear insights into feature importance. It is well-suited for imbalanced classification when combined with class weighting or resampling.
- **Ensemble Model (Random Forest or Gradient Boosting):** Chosen for their ability to capture complex, non-linear relationships and handle feature interactions, which are common in fraud detection. These models are robust to outliers and can provide feature importance rankings.

Model evaluation will focus on metrics appropriate for imbalanced data, such as F1-score and AUC-PR. The choice of models aligns with the business need for both explainability (to build trust with stakeholders) and high predictive performance (to minimize financial losses).

## Handling Class Imbalance
Class imbalance is a significant challenge in fraud detection. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data:

- **Before SMOTE:** The dataset is highly imbalanced, with very few fraud cases compared to legitimate transactions.
- **After SMOTE:** The training set is balanced, allowing the model to better learn fraud patterns and generalize to new, unseen cases.

SMOTE was chosen because it creates synthetic samples of the minority class, leading to a more informative training set and improved model generalization. The effectiveness of SMOTE will be evaluated using appropriate metrics for imbalanced data, such as F1-score and AUC-PR.

## EDA & Feature Engineering Insights
- **Fraud Patterns:**
  - The dataset is highly imbalanced, with fraudulent transactions representing a small fraction of the total
  - Fraudulent transactions tend to have higher purchase values
  - Certain sources and browsers are associated with higher fraud rates
  - Temporal and geolocation patterns reveal additional risk factors
- **Feature Engineering:**
  - Time-based features (hour_of_day, day_of_week, time_since_signup) capture behavioral signals
  - Transaction frequency and velocity features help identify suspicious activity
  - All features are chosen based on domain knowledge and observed data patterns

## Class Imbalance Strategy
- **Challenge:** Fraudulent transactions are rare, making standard modeling approaches ineffective
- **Solution:** SMOTE is applied to the training data to generate synthetic fraud cases, improving the model's ability to detect fraud
- **Justification:** This approach increases recall and precision for the minority class, which is critical for business impact

## Processed Data

- All cleaned and feature-engineered datasets are saved in `data/processed/`.
- These files are ready for use in model training and evaluation in subsequent tasks.

## Key Deliverables for First Interim Submission
- Data cleaning and preprocessing (missing values, duplicates, type corrections)
- Exploratory Data Analysis (EDA) with business-relevant insights and visualizations
- Feature engineering with clear business hypotheses
- Geolocation analysis (IP to country mapping)
- Handling class imbalance using SMOTE (requires `imbalanced-learn`)
- Well-documented notebook with markdown explanations and conclusions
- These files are ready for use in model training and evaluation in subsequent tasks

## Running in Google Colab
- Mount your Google Drive and adjust data paths as needed
- All dependencies are listed in `requirements.txt` (including `imbalanced-learn` for SMOTE)

## Business Context & Rubric Alignment
- The project is structured to meet business needs for fraud detection, with a focus on actionable insights, explainability, and reproducibility
- All documentation and code are aligned with the grading rubric and KAIM instructions for the interim submission
- Each deliverable is mapped to rubric criteria for maximum clarity and transparency

## Documentation and Collaboration
- All code is thoroughly commented and explained, especially in the feature engineering and class imbalance sections.
- The notebook and this README provide clear guidance for reviewers and collaborators.
- For further details, see the Jupyter notebook (`notebooks/01_task1_data_cleaning_eda.ipynb`).
- Contributions are welcome; please fork the repo and submit a pull request.

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [KAIM Challenge Guidelines]

## Contributing, Author, License
- **Contributions:** Welcome! Please fork the repo and submit a pull request
- **Author:** Tinbite Yonas
- **License:** [Specify license here]
