from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import classification_report



def evaluate_bayesian_model(X_test, y_test, bayesian_network_model):
    y_pred = bayesian_network_model.predict(X_test)
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    confusion = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))
    f,ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusion, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




def evaluate_neural_model(model, X_test, y_test, df):
    predictions = model.predict(X_test)
    predictions = np.round(predictions).flatten()
    accuracy = accuracy_score(y_test, predictions)
    print(type(y_test))
    cfm = confusion_matrix(y_test, predictions)
    print(accuracy)

    print(classification_report(y_test, predictions))
    f,ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
