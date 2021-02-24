from xgboost import XGBClassifier
#https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
import numpy as np
import pandas as pd
# https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# https://www.freecodecamp.org/news/how-to-use-the-tree-based-algorithm-for-machine-learning/
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def xgboost(x_test, x_train, y_train, y_test, directory, day: int):
    resultFile = open(f"xgboost-{directory}-{day}-result.txt", "a")
    model = XGBClassifier()
    model.fit(x_train, y_train)
    # make predictions for test data
    y_pred = model.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    Accuracy = accuracy_score(y_test, y_pred)
    resultFile.write(format(confusion_matrix(y_test, y_pred)))
    resultFile.write(format(classification_report(y_test, y_pred)))
    return y_pred, precision, recall, Accuracy
