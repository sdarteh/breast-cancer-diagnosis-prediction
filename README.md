# Breast Cancer Diagnosis Prediction Using Classification

## Project Overview

This project applies machine learning to breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) dataset. It evaluates 13 linear and non-linear classification models, compares their predictive performance, and identifies key diagnostic predictors to select the best-performing model.

## Abstract

<blockquote><small>
Breast cancer, undoubtedly, is a significant global health concern; it is today’s commonest cancer amongst women, and the second to lung cancer in mortality rate. Early and accurate diagnosis is a critical resolve for effective prevention, treatment, and improved patient outcomes. This study explores the application of predictive modeling techniques to enhance breast cancer diagnosis. A comprehensive dataset comprising clinical imaging and genetic features from breast mass sampling is used to train and evaluate predictive models—classification models. Machine learning algorithms deployed in both linear and non-linear classification models including neural networks, regularized discriminant analysis, and penalized models are utilized to analyze the dataset and predict the likelihood of breast cancer occurrence. The performance of these models, in both training and test, is assessed based on Kappa statistic, accuracy rate, and ROC’s AUC (a balance for sensitivity and specificity). Lastly, feature importance analysis is conducted to identify predictors that are most critical to the optimal predictive model’s outcomes.
</small></blockquote>

## Research Objectives

* Develop machine learning models to predict breast cancer diagnosis as benign or malignant.
* Compare the performance of linear and non-linear classification techniques.
* Evaluate model performance using accuracy, Kappa statistic, and ROC-AUC.
* Identify the best-performing models using training results and evaluate their performance on unseen test data.
* Determine the most influential predictors contributing to the selected model's predictions.

## Dataset

The study uses the **Breast Cancer Wisconsin (Diagnostic) dataset** from the University of California, Irvine (UCI).

The dataset contains 569 observations. Its diagnostic features were computed from digitized images of fine needle aspirate samples of breast masses.

The original dataset includes 30 numerical predictors derived from 10 characteristics of cell nuclei, with each characteristic represented by mean, standard error, and worst-value measurements.

The target variable, `diagnosis`, contains two classes:

* **Benign (B):** Non-malignant diagnosis.
* **Malignant (M):** Malignant diagnosis.

**Dataset source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Methodology

### 1. Data Preprocessing

The preprocessing stage involved:

* Exploring the dataset's structure and variables.
* Assigning descriptive names to the dataset's columns.
* Encoding the diagnosis variable as a two-class factor: Benign (B) and Malignant (M).
* Examining correlations among predictors and considering feature-reduction approaches.
* Transforming and preparing the data for model training.
* Partitioning the 569 observations into training and test datasets.

The dataset was partitioned as follows:

| Dataset   | Observations | Percentage |
| --------- | -----------: | ---------: |
| Training  |          456 |        80% |
| Testing   |          113 |        20% |
| **Total** |      **569** |   **100%** |

The study used stratified random sampling and repeated 10-fold cross-validation, repeated five times, during model development.

### 2. Classification Models

A total of **13 classification models** were trained and evaluated.

#### Linear Classification Models

1. Logistic Regression
2. Linear Discriminant Analysis (LDA)
3. Partial Least Squares Discriminant Analysis (PLSDA)
4. Penalized Model
5. Nearest Shrunken Centroids (NSC)

#### Non-Linear Classification Models

6. Quadratic Discriminant Analysis (QDA)
7. Regularized Discriminant Analysis (RDA)
8. Mixture Discriminant Analysis (MDA)
9. Flexible Discriminant Analysis (FDA)
10. Neural Networks (NNet)
11. Support Vector Machines (SVM)
12. K-Nearest Neighbors (KNN)
13. Naive Bayes (NB)

Model tuning was performed where applicable to identify suitable model parameters.

### 3. Model Evaluation

The models were evaluated using the following performance metrics:

| Metric          | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| Accuracy        | Measures the proportion of observations correctly classified.                         |
| Kappa Statistic | Measures agreement between predicted and actual classifications, adjusted for chance. |
| ROC-AUC         | Measures the model's ability to distinguish between the two diagnostic classes.       |

Kappa statistic was used as the primary criterion for comparing models during training, followed by accuracy.

The two leading models selected for further evaluation on unseen test data were:

* Regularized Discriminant Analysis (RDA)
* Penalized Model

### 4. Feature Importance Analysis

Feature importance analysis was performed using the selected Penalized Model to identify the predictors most influential in its predictions.

The five highest-ranked predictors reported in the study were:

| Rank | Predictor             | Relative Importance |
| ---: | --------------------- | ------------------: |
|    1 | `concave_points_mean` |             100.000 |
|    2 | `radius_mean`         |              81.691 |
|    3 | `texture_worst`       |              78.984 |
|    4 | `radius_error`        |              72.602 |
|    5 | `symmetry_worst`      |              43.768 |

These values represent the relative importance scores reported by the study, not diagnostic probabilities.

## Results and Findings

The study trained all 13 classification models and evaluated their performance. The two leading models from training—RDA and the Penalized Model—were subsequently evaluated using the unseen test dataset.

The reported test results were:

| Metric          | Penalized Model |    RDA |
| --------------- | --------------: | -----: |
| Accuracy        |          96.46% | 96.46% |
| Kappa Statistic |          92.57% | 92.57% |
| ROC-AUC         |          99.77% | 99.66% |

## Conclusion and Recommended Model

All 13 classification models were trained, and the two leading models were evaluated on unseen test data.

Although both the Penalized Model and Regularized Discriminant Analysis (RDA) achieved a reported test accuracy of 96.46%, the Penalized Model achieved a slightly higher ROC-AUC of 99.77%, compared with 99.66% for RDA.

**The Penalized Model was selected as the recommended model in the study.**

The study demonstrates how machine learning classification techniques can be applied to diagnostic data, compared using established performance metrics, and examined through feature importance analysis.

Further validation using appropriate independent datasets and clinical evaluation would be necessary before considering any model for real-world clinical use.

## Repository Contents

This repository contains the research project materials:

| File or Directory     | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `README.md`           | Project overview, methodology, models, results, and conclusions.       |
| `preproc.R`           | R code for data preprocessing and preparation.                         |
| `linearmodels.R`      | R code for the linear classification models.                           |
| `non-linearmodels.R`  | R code for the non-linear classification models.                       |
| Research report (PDF) | Full research paper, methodology, experiments, and supporting results. |

## Technologies and Techniques

* **Programming Language:** R
* **Machine Learning:** Classification and predictive modeling
* **Model Evaluation:** Accuracy, Kappa statistic, and ROC-AUC
* **Data Preparation:** Data transformation, predictor analysis, and dataset partitioning
* **Validation:** Stratified sampling and repeated 10-fold cross-validation
* **Feature Analysis:** Predictor importance ranking

## Author

**[Samuel Darteh](https://linkedin.com/in/sdartehpro)** <br> 
LinkedIn: (https://linkedin.com/in/sdartehpro) <br>
[Michigan Technological University](https://mtu.edu) | MS Data Science


## Disclaimer

This project is presented for academic and research purposes. The models are not intended to replace professional medical judgment or serve as standalone clinical diagnostic tools.
