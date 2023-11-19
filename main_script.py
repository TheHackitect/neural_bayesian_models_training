# Import necessary libraries
import data_preprocessing as dp
import model_creation as mc
import evaluation as ev
from model_creation import split_dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = dp.load_data('survey.csv')

# Data cleaning and preprocessing
df = dp.clean_and_preprocess(df)

print(df)
print(df.info())


for column in df.select_dtypes(include=['object']).columns:
    # Count the number of unique categories in the column
    num_categories = df[column].nunique()
    print(f'The column "{column}" has {num_categories} unique categories.')



le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

print(df)

X_train, X_test, y_train, y_test = split_dataset(df)

# Create Bayesian Network model
bayesian_network_model = mc.create_bayesian_model(X_train,y_train)

# Create Neural Network model
neural_network_model = mc.create_neural_network_model(X_train, y_train, X_test, y_test)

# Evaluate models
bayesian_metrics = ev.evaluate_bayesian_model(X_test, y_test, bayesian_network_model)
neural_network_metrics = ev.evaluate_neural_model(neural_network_model, X_test, y_test, df)
