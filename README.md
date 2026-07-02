# Autism Spectrum Disorder (ASD) Screening Prediction

This repository contains the coursework project for **BCSE109L - Machine Learning**. The project focuses on developing a machine learning model to predict the likelihood of an individual having Autism Spectrum Disorder based on responses to a screening questionnaire and demographic data.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Web Application](#web-application)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Project Overview

The primary goal of this project is to explore various machine learning techniques to build an effective ASD screening tool. The process involves comprehensive data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation. Several advanced methods like ensemble learning, meta-learning, and hybrid models are investigated to achieve the best possible performance on an imbalanced dataset.

## Project Structure

```
.
├── App/
│   └── app.py                # Streamlit web application
├── Data/
│   ├── Processed Data/       # Processed and encoded datasets
│   └── Raw Data/             # The original raw dataset
├── Models/
│   └── best_model.joblib     # The final, best-performing trained model
├── results graphs/
│   └── *.png                 # Various plots and graphs from the analysis
├── BaseModels.ipynb          # EDA and training of baseline models (SVM, KNN, etc.)
├── Best_Model.ipynb          # In-depth evaluation of the final selected model
├── Dataprocessing.ipynb      # Data cleaning, preprocessing, and feature engineering
├── Ensemble.ipynb            # Training and evaluation of ensemble models (RF, AdaBoost, XGBoost)
├── Hybrid_Ensemble.ipynb     # Implementation of a custom hybrid ensemble model
├── Hyper_Parameter.ipynb     # Hyperparameter tuning to find the best model and parameters
├── Meta_Learning.ipynb       # Implementation of a meta-learning (stacking) model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Methodology

1.  **Data Preprocessing**: The initial dataset was cleaned by handling missing values, correcting data types, and removing irrelevant columns. Outliers in the `age` feature were managed by replacing them with the median value. The `contry_of_res` feature was mapped to a more general `region` feature. This is detailed in [`Dataprocessing.ipynb`](Dataprocessing.ipynb).

2.  **Feature Encoding**: Categorical features were encoded using three different strategies to evaluate their impact on model performance:
    *   **Target Encoding**
    *   **Frequency Encoding**
    *   **One-Hot Encoding**

3.  **Handling Class Imbalance**: The dataset is imbalanced. Techniques like **SMOTE** and **ADASYN** were applied to the training data to create a more balanced class distribution, which is crucial for training unbiased models.

4.  **Modeling**: A wide range of models were trained and evaluated:
    *   **Base Models**: SVM, KNN, Decision Tree ([`BaseModels.ipynb`](BaseModels.ipynb)).
    *   **Ensemble Models**: Random Forest, AdaBoost, XGBoost ([`Ensemble.ipynb`](Ensemble.ipynb)).
    *   **Meta-Learning**: A stacking classifier using base models to train a final logistic regression meta-learner ([`Meta_Learning.ipynb`](Meta_Learning.ipynb)).
    *   **Hybrid Ensemble**: A custom model that combines meta-features from base models with the most important original features, trained with an MLP classifier ([`Hybrid_Ensemble.ipynb`](Hybrid_Ensemble.ipynb)).

5.  **Hyperparameter Tuning**: `GridSearchCV` was used to systematically search for the optimal hyperparameters for the most promising models (SVM, AdaBoost, XGBoost) across different data encodings and SMOTE applications ([`Hyper_Parameter.ipynb`](Hyper_Parameter.ipynb)).

6.  **Evaluation**: The primary metric for model evaluation was the **F1-Score**, which is well-suited for imbalanced classification tasks. Other metrics like Precision, Recall, Accuracy, and AUC were also used for a comprehensive analysis.

## Results

After extensive experimentation, the best-performing model was identified through the hyperparameter tuning process in [`Hyper_Parameter.ipynb`](Hyper_Parameter.ipynb).

-   **Best Model**: Support Vector Classifier (SVC)
-   **Best Encoding**: Target Encoding
-   **SMOTE**: Not applied (performed better on the original imbalanced data)
-   **Test F1-Score**: **0.743**

The final model and its configuration were saved to `Models/best_model.joblib`. A detailed performance analysis, including confusion matrix, ROC curve, and precision-recall curve, is available in [`Best_Model.ipynb`](Best_Model.ipynb).

## Web Application

A user-friendly web application was developed using Streamlit to provide an interactive interface for the screening tool. Users can answer the AQ-10 screening questions and provide demographic information to get a real-time prediction.

The application is implemented in [`App/app.py`](App/app.py).

## How to Run

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```sh
    streamlit run App/app.py
    ```

## Dependencies

Install the Python dependencies listed in [`requirements.txt`](requirements.txt).