# PREDICTING-HOUSE-PRICES-WITH-MACHINE-LEARNING

# Project Title: House Price Prediction

# Description:
This project focuses on predicting house prices using the Boston Housing Price Dataset and the XGBoost Regressor model. The code is written in Python and utilizes libraries such as NumPy, Pandas, scikit-learn, and XGBoost.

# Key Features:
Loads the Boston Housing Price Dataset.    
Preprocesses data using Pandas DataFrames.    
Trains an XGBoost Regressor model.    
Evaluates model performance.    
Visualizes actual vs. predicted prices.    
Code Highlights:

# Importing Dependencies:
Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
   

# Loading the Dataset:
Python

house_price_dataset = sklearn.datasets.load_boston()
   

# Creating Pandas DataFrame:
Python

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
   

# Training the XGBoost Model:
Python

model = XGBRegressor()
model.fit(x_train, y_train)
   

# Evaluating Model Performance:
Python

training_data_prediction = model.predict(x_train)
score_1 = metrics.r2_score(y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)
print("R squared error : ", score_1)
print("Mean Absolute error : ", score_2)
   

# Installation:
Make sure you have Python 3.x installed.

Install the required libraries:
Bash

pip install numpy pandas scikit-learn xgboost matplotlib seaborn

# Usage:
Clone the repository.
Run the Python script.

# Future Improvements:
Explore feature engineering techniques.
Fine-tune the XGBoost model parameters.
Evaluate the model on other housing datasets.
Implement cross-validation.
