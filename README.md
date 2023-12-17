# Project Title: Home Credit Default Risk

## Abstract
In response to Home Credit's challenge of assessing creditworthiness for clients with limited credit history, our project employs Logistic Regression with Lasso regularization (LASSO-CXE) and the K-Nearest Neighbors (KNN) algorithm.

We tackle data challenges through advanced techniques such as data cleaning, feature engineering, and the creation of new features. To address imbalanced datasets, we evaluate model performance using key metrics such as ROC AUC, F1 Score, and Balanced Accuracy. These metrics provide a nuanced understanding of the classifier's performance, considering both false positives and negatives.

Our goal is to enhance Home Credit's lending decisions, reduce unpaid loans, and extend financial services to individuals with limited access to traditional banking. The Logistic Regression model with Lasso regularization aids in feature selection and prevents overfitting, while KNN's adaptability proves valuable in assessing credit risk by identifying patterns in borrower profiles. This comprehensive approach ensures the development of a robust model for effective credit risk assessment.

## Introduction
### Background on Home Credit
Home Credit, a non-banking financial institution established in 1997 in the Czech Republic, caters to individuals with limited or no credit history who might otherwise be denied loans or fall prey to unscrupulous lenders. Operating in 14 countries, including the United States, Russia, Kazakhstan, Belarus, China, and India, Home Credit has amassed over 29 million customers, granted over 160 million loans, and accumulated total assets of 21 billion euros, with the majority of its business located in Asia, particularly China (as of May 19, 2018).

Currently employing various statistical and machine learning techniques to assess creditworthiness, Home Credit seeks Kagglers' assistance in unlocking the full potential of their data. This endeavor aims to ensure that creditworthy clients are not overlooked and that loans are tailored with appropriate principal amounts, maturities, and repayment schedules to empower clients' financial success.

### Data Description
The Home Credit Default Risk dataset, obtained from the Kaggle project, aims to help Home Credit make informed decisions about loan applications for individuals who may not qualify through traditional banking systems. To accomplish this, Home Credit gathers various data sources, including phone and transaction records, to evaluate a borrower's ability to repay a loan.

At the heart of this dataset is the "application {train test}" table, which contains the loan applications that will be analyzed for potential default risk. Six additional tables provide supplementary information related to the primary table, forming a hierarchical structure. Detailed explanations of these tables are available from the HCDR Kaggle Competition.

### Data files overview
POS_CASH_balance.csv --> Shape: (10001358, 8) --> Numerical Features: 7 --> Categorical Features: 1
application_test.csv --> Shape: (48744, 121) --> Numerical Features: 105 --> Categorical Features: 16
application_train.csv --> Shape: (307511, 122) --> Numerical Features: 106 --> Categorical Features: 16
installments_payments.csv --> Shape: (13605401, 8) --> Numerical Features: 8 --> Categorical Features: 0
bureau_balance.csv --> Shape: (27299925, 3) --> Numerical Features: 2 --> Categorical Features: 1
credit_card_balance.csv --> Shape: (3840312, 23) --> Numerical Features: 22 --> Categorical Features: 1
installments_payments.csv --> Shape: (13605401, 8) --> Numerical Features: 8 --> Categorical Features: 0
bureau.csv --> Shape: (1716428, 17) --> Numerical Features: 14 --> Categorical Features: 3



application_{train|test}.csv: This table contains static data for loan applications. The "train" version includes a target variable, while the "test" version does not.

bureau.csv: It holds information about a client's previous credits from other financial institutions reported to the Credit Bureau. Multiple rows can correspond to a single loan application.

bureau_balance.csv: This table provides monthly balances of previous credits reported to the Credit Bureau, creating multiple rows for each loan's history.

POS_CASH_balance.csv: It contains monthly snapshots of the balance for point of sales and cash loans that the applicant had with Home Credit, generating multiple rows for each loan's history.

credit_card_balance.csv: This table shows monthly balance snapshots of previous credit cards the applicant had with Home Credit, with multiple rows for each card's history.

previous_application.csv: This dataset includes all previous loan applications made by clients in the sample, with one row per application.

installments_payments.csv: It covers repayment history for credits disbursed by Home Credit, with one row for each payment or missed payment.

HomeCredit_columns_description.csv: This file provides descriptions for the columns in the various data files, helping users understand the data better.

### Data Dictionary
The data download includes a Data Dictionary named HomeCredit_columns_description.csv. This file provides detailed information about all the fields present in the accompanying data tables. In other words, it serves as a comprehensive metadata resource for the entire dataset.

**The project is developed in four phases. Each phase having it's importance. The below is summary of Final Phase**

## Tasks to be tackled
The tasks to be addressed in this phase of the project are given below:

Join the datasets : Consolidate the remaining datasets into a unified dataset that encompasses all pertinent customer information.

Perform EDA on other datasets : Perform Exploratory Data Analysis (EDA) on the individual datasets, excluding the application_train dataset and the merged datasets, to uncover patterns, trends, and relationships among the various data attributes.

Identify missing values and highly correlated features in the merged data : Identify and address missing values within the merged dataset. Additionally, eliminate highly correlated features to mitigate the risk of multicollinearity.

Detect and mitigate potential errors in the merged data : Scrutinize the merged dataset for any errors that could potentially impact the model's performance. Implement appropriate measures to rectify these errors and ensure data integrity.

Incorporate domain knowledge features: Incorporate domain-specific features that have the potential to improve the model's predictive capabilities.

Analyze the impact of newly added features on the target variable: Analyze the correlation between the newly introduced features and the target variable to assess their impact on the model's predictive accuracy.

Build upon models from Phase 2: Augment the existing models from Phase 2, particularly Logistic Regression, by incorporating the newly extracted features and insights gained from the current phase to enhance their predictive capabilities.

Model selection and training: Select appropriate machine learning algorithms, including lasso regression, logistic regression, decision trees, random forests, gradient boosting machines (GBMs), and neural networks. Divide the data into training and testing sets and utilize the training data to train the chosen models.

Calculate and validate the results: Assess the performance of the refined models employing pertinent evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Conduct thorough validation to verify the models' efficacy in predicting default probabilities.

Model evaluation: To assess the effectiveness of the developed models, we will evaluate their performance using relevant metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. By comparing the performance of these models based on these evaluation metrics, we will identify the model that demonstrates the strongest predictive capabilities.

Perform hyperparameter tuning with GridSearchCV: Employ GridSearchCV to identify the optimal hyperparameters for the selected models and enhance their predictive performance.

Perform ensemble modeling: Leverage ensemble modeling techniques to enhance the predictive capabilities of the developed models potentially.

The implementation of the most effective predictive model will empower Home Credit to make informed lending decisions, reduce the likelihood of unpaid loans, and expand financial services to individuals with limited access to traditional banking, thereby promoting financial inclusion for underserved communities. The performance of our models in predicting default probabilities will be rigorously evaluated using key metrics such as ROC AUC and F1 Score. Additionally, we will assess both public and private scores to gain a comprehensive understanding of our model's efficacy.


## Pipelines Implemented (Phase 4)
Families of input features:
Count of numerical features: 107
Count of categorical features: 16
The total number of input features: 124 input features with target.
We have below trained Three MLP models :
- Simple Multi-Layer Perceptron (MLP)
- PyTorch implementation on MLP (enhanced MLP)
- Deep Wider MLP architecture

### Data Leakage
Data leakage occurs when a model is trained on information that will not be available during the prediction phase, resulting in artificially inflated performance metrics. To prevent data leakage, the dataset should be split into training and testing sets before any data preprocessing is performed. Missing values should be handled and standardization should be applied to each set independently. By training the model on the training set and transforming the testing set using the same method, we can ensure that the model's performance accurately reflects its real-world capabilities.

### Cardinal Sins avoided:
- Our machine learning pipelines adhere to best practices and avoid the cardinal sins of machine learning:

- Overfitting Prevention: We split our dataset into training and testing sets to prevent overfitting. The model is trained on the training set and evaluated on the unseen test set. Similar accuracy on both sets indicates that the model is not overfitting.

- Convergence Monitoring: We monitor training progress using Tensorboard graphs and avoid arbitrarily increasing epochs. We only increase epochs when the loss curve indicates a high learning rate, ensuring convergence.

- Balanced Dataset: We ensure a balanced dataset to accurately evaluate model performance using metrics like accuracy and ROC_AUC.

- Accurate Labels: We employ accurate labels in the training dataset to ensure the model learns from reliable information.

- These measures safeguard against common pitfalls and ensure the effectiveness of our machine learning models.

### Loss Function used:
The binary cross-entropy loss function will be utilized by this MLP class.

**Refer to each Phase notebook for detailed reports and result discussion.**


This Project is contributed by Sai Sathwik Reddy Varikoti, Bindu Madhavi Dokala, Jagadeesh Kovi, and Pranay Chowdary Namburi.
