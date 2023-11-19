from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyAgrum.skbn import BNClassifier

import pandas as pd
import data_preprocessing as dp
import joblib
import numpy as np




def split_dataset(df):
    X = df.drop('treatment', axis=1)

    y = df['treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



def create_bayesian_model(X_train,y_train):    
    model2=BNClassifier()
    model2.fit(X_train,y_train)
    return(model2)


def create_neural_network_model(X_train, y_train, X_test, y_test):
    # Create and train the Neural Network model using Keras
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Save the model to a file
    save_model(model, 'neural_network_model.keras')
    plot_model(model, 'my_first_model.png', show_shapes=True)

    return(model)
