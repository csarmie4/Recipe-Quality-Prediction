# Recipe Score Prediction

This project uses logistic regression and random forest models to classify recipes as high or low scoring based on nutritional and categorical data. It includes data loading, preprocessing, visualization, model training, and evaluation steps.

## Table of Contents
- [Project Overview](#project-overview)
- [Scripts and Modules](#scripts-and-modules)
- [Results](#results)
- [Requirements](#requirements)
- [Setup and Execution](#setup-and-execution)
- [Contributors](#contributors)

## Project Overview
This project implements a machine learning pipeline for predicting recipe scores. The main steps include:
1. **Data Loading**: Loading recipe data from a CSV file.
2. **Data Preprocessing**: Cleaning and transforming data for modeling.
3. **Data Visualization**: Generating correlation heatmaps and distribution plots.
4. **Model Training**: Training logistic regression and random forest models.
5. **Model Evaluation**: Evaluating model performance using confusion matrices and classification reports.

## Scripts and Modules

- **`main.py`**: Coordinates data loading, visualization, model training, and evaluation.
- **`data_preprocessing.py`**: Contains functions for loading and preprocessing data.
- **`visualization.py`**: Provides functions for plotting data correlations, high-score distributions, box plots, and QQ plots.
- **`model_training.py`**: Includes functions for training logistic regression and random forest models, with oversampling to handle class imbalance.
- **`model_evaluation.py`**: Contains functions to evaluate trained models and print detailed performance metrics.
- **`__init__.py`**: Initializes the modules in the package.

## Results

### Logistic Regression
- **Training Accuracy**: 0.53
- **Test Accuracy**: 0.53

#### Confusion Matrix
[[2766 2673] [2541 3081]]
- **Accuracy on Low Scoring Recipes**: 52.12%

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.52      | 0.51   | 0.51     | 5439    |
| 1     | 0.54      | 0.55   | 0.54     | 5622    |
| **Accuracy** | | | 0.53 | 11061 |
| **Macro Avg** | 0.53 | 0.53 | 0.53 | 11061 |
| **Weighted Avg** | 0.53 | 0.53 | 0.53 | 11061 |

### Random Forest
- **Training Accuracy**: 1.00
- **Test Accuracy**: 0.75

#### Confusion Matrix
[[4189 1250] [1568 4054]]
#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.73      | 0.77   | 0.75     | 5439    |
| 1     | 0.76      | 0.72   | 0.74     | 5622    |
| **Accuracy** | | | 0.75 | 11061 |
| **Macro Avg** | 0.75 | 0.75 | 0.75 | 11061 |
| **Weighted Avg** | 0.75 | 0.75 | 0.75 | 11061 |

## Performance Metrics

### Accuracy
- **Definition**: The ratio of correct predictions to the total number of predictions.
- **Importance**: While accuracy provides a general sense of model performance, it can be misleading in cases of class imbalance, where one class dominates, as in this dataset.

### Precision
- **Definition**: The ratio of true positive predictions to the total positive predictions (true positives + false positives).
- **Importance**: Precision is crucial when the cost of false positives is high. For example, predicting a low-quality recipe as high quality could negatively impact user experience.

### Recall
- **Definition**: The ratio of true positive predictions to the total actual positives (true positives + false negatives).
- **Importance**: Recall is key when the cost of false negatives is high. In this context, it indicates the model's ability to capture all high-scoring recipes.

### F1 Score
- **Definition**: The harmonic mean of precision and recall.
- **Importance**: The F1 score balances precision and recall, providing a single metric for understanding overall model effectiveness, especially when there is class imbalance.

### Confusion Matrix
- **Definition**: A table showing true positives, true negatives, false positives, and false negatives.
- **Importance**: The confusion matrix helps visualize model performance across all classes, showing strengths and weaknesses in specific predictions.

## Requirements
- Python 3.x
- Required packages are listed in `requirements.txt`.

