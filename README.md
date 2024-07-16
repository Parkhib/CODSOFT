CREDIT CARD FAULT DETECTION SYSTEM
Credit Card Fraud Detection using Logistic Regression on credit card dataset
As this is a binary classification problem, we will be using Logistic Regression model for model training.

WORKFLOW OF MODEL
1.Collection of data
2.Data Preprocessing
3.Splitting test and training data
4.Model Training
5.Model Evaluation
6.Prediction System

DEPENDECIES USED:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET
The dataset used for this research is collected from Kaggle at https://www.kaggle.com/mlg-ulb/creditcardfraud. It consists of 284,807 transactions that occurred in 2 days, of which 492 are labelled as Fraud. This means that the dataset is highly unbalanced with only 0.172% accounting for the Fraud transactions. It consists of 31 features of which 28 (V1-V28) are the result of PCA transformation, due to confidentiality issues. The remaining features that are not transformed are ‘Time’ and ‘Amount’, which represent the seconds elapsed between each transaction and the first transaction in the dataset and, the transaction amount respectively. The ‘Class’ feature represents the label of the transaction with ‘1’ for a Fraud transaction and ‘0’ for a ‘Genuine’ transaction.

[Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015]

DATA ANALYSIS
1.shape
2.info()
3.describe()
4.isnull
5.count_values()
6.dtypes

SAMPLING THE DATA
0 : Normal transaction
1 : Fraudulent transaction

UNDER-SAMPLING THE DATA
build a sample dataset having similar distribution of normal and fraudulent transactions.
number of fraudulent transaction is = 492.

MODEL EVALUATION
print("\nAccuracy on Training data ",traning_data_accuracy,"\n")
print("Accuracy on Training data ",test_data_accuracy)

Accuracy on training data = Accuracy :  0.9351969504447268
Accuracy on testing data = Accuracy : 0.9137055837563451.

