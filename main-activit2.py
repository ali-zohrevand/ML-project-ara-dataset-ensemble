# from ensemble import branching
# for directory in range(2):
#     print(f"directory {directory+1}")
#     for activity in range(2):
#         print(f"activity {activity}")
#         branching(f'House{directory + 1}', f'activity{activity + 1}', "xgboost.md")
#
#
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from boost import xgboost
from knn import knn
from logistic import lr
from bernoullinb import bernoullinb

columnsString = "daySection,Ph1,Ph2,Ir1,Fo1,Fo2,Di3,Di4,Ph3,Ph4,Ph5,Ph6,Co1,Co2,Co3,So1,So2,Di1,Di2,Te1,Fo3,activity1,activity2"
col_names = columnsString.split(",")
col_names = columnsString.split(",")
dataset = pd.read_csv(f"data-House1.csv", header=None, names=col_names)
X = dataset.drop('activity2', axis=1)
y = dataset['activity2']
yArray = np.array(y)
xArray = np.array(X)
xArray = xArray.astype(np.float)
yArray = yArray.astype(np.float)
print("=============================\n")
print("knn")
X_train, X_test, y_train, y_test = train_test_split(xArray, yArray, test_size=0.20)
# knn_y_pred, knn_precision, knn_recall, knn_Accuracy = knn(X_test, X_train, y_train, y_test, 'House1', 'AllDay')
print("bernoullinb\n")

bernoullinb_y_pred, bernoullinb_precision, bernoullinb_recall, bernoullinb_Accuracy = bernoullinb(X_test, X_train,
                                                                                                  y_train, y_test,
                                                                                                  'House1', 'AllDay')
print("LogisticRegression\n")

lr_y_pred, lr_precision, lr_recall, lr_Accuracy = lr(X_test, X_train, y_train, y_test, 'House1', 'AllDay')
# DecisionTreeClassifier_y_pred, DecisionTreeClassifier_precision, DecisionTreeClassifier_recall, DecisionTreeClassifier_Accuracy = DecisionTreeClassifier(X_test, X_train, y_train, y_test, 'House1', 'AllDay')
print("xgboost\n")
xgboost_y_pred, xgboost_precision, xgboost_recall, xgboost_Accuracy = xgboost(X_test, X_train, y_train, y_test,
                                                                              'House1', 'AllDay')
selected_array = []
for i in range(len(y_test)):
    max_vote = 0
    selected_value = xgboost_y_pred[i]
    predicted = [xgboost_y_pred[i], lr_y_pred[i], bernoullinb_y_pred[i]]
    for p in predicted:
        counter = 0
        for obj in predicted:
            if obj == p:
                counter = counter + 1
        if counter> max_vote:
            max_vote=counter
            selected_value=p

    selected_array.append(selected_value)
selected_array = np.array(selected_array)
precision = precision_score(y_test, selected_array, average='macro')
recall = recall_score(y_test, selected_array, average='macro')
Accuracy = accuracy_score(y_test, selected_array)
print("=============================\n")

print("precision")
print(precision)
print("=============================\n")

print("Recall")
print(recall)
print("=============================\n")

print("Accuracy")
print(Accuracy)
