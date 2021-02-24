import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from boost import xgboost
def predict(directory,activity):
    resultFile = open("result", "a")
    resultFile.write(f"===========================\n{directory}  {activity}\n")

def branching(directory, activity, output):
    resultFile = open(output, "a")
    earlyMorning = 18000
    morning = 43200
    noon = 57600
    night = 72000
    totalAccuracy=0
    resultFile.write(f"===========================\n{directory}  {activity}\n")
    resualt = "     precision        recall        accuracy\n"
    for day in range(30):
        i: int = 0
        f = open(f"{directory}/DAY_{day + 1}.txt", "r")
        output = ""
        for x in f:
            daySection = 0
            if i > night:
                daySection = 4
            if earlyMorning < i <= morning:
                daySection = 1
            if morning < i <= noon:
                daySection = 2
            if noon < i <= night:
                daySection = 3
            line = x.replace(" ", ",")
            line = str(daySection) + "," + line
            output = output + line
            i = i + 1
        f = open(f"{directory}/DAY_{day + 1}.csv", "a")
        f.write(output)
        f.close()
        print("=============================\nDay: " + str(day + 1))
        pima = pd.read_csv(f"{directory}/DAY_{day + 1}.csv")
        columnsString = "daySection,Ph1,Ph2,Ir1,Fo1,Fo2,Di3,Di4,Ph3,Ph4,Ph5,Ph6,Co1,Co2,Co3,So1,So2,Di1,Di2,Te1,Fo3,activity1,activity2"
        col_names = columnsString.split(",")
        dataset = pd.read_csv(f"{directory}/DAY_{day + 1}.csv", header=None, names=col_names)
        X = dataset.drop('activity1', axis=1)
        X = dataset.drop('activity2', axis=1)
        y = dataset[activity]
        yArray = np.array(y)
        xArray = np.array(X)
        xArray = xArray.astype(np.float)
        yArray = yArray.astype(np.float)
        X_train, X_test, y_train, y_test = train_test_split(xArray, yArray, test_size=0.20)
        dtree_y_pred, dtree_precision, dtree_recall, dtree_Accuracy = xgboost(X_test, X_train,y_train, y_test, directory, day)
        resualt=f"xgboost {day+1}       {dtree_precision}       {dtree_recall}      {dtree_Accuracy}\n"
        totalAccuracy=totalAccuracy+dtree_Accuracy
        resultFile.write(resualt)
    totalAccuracy=totalAccuracy/30
    resultFile.write(f"\n total accuracy {directory} is: {totalAccuracy} ")
    resultFile.close()