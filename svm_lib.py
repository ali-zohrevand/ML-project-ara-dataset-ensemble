import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
#https://www.freecodecamp.org/news/how-to-use-the-tree-based-algorithm-for-machine-learning/
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import rcParams
import warnings
def svm_function(x_test, x_train, y_train, y_test, directory, day: int):
    resultFile = open(f"svm-{directory}-{day}-result.txt", "a")
    clf = svm.SVC(kernel='linear')
    # Linear Kernel
    #Train the model using the training sets
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    Accuracy = accuracy_score(y_test, y_pred)
    resultFile.write(format(confusion_matrix(y_test, y_pred)))
    resultFile.write(format(classification_report(y_test, y_pred)))
    return y_pred, precision, recall, Accuracy
