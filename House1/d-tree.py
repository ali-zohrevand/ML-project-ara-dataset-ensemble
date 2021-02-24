import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def decision_tree_house(directory, activity, output):
    resultFile = open(output, "a")
    earlyMorning = 18000
    morning = 43200
    noon = 57600
    night = 72000
    resultFile.write(f"{directory}  {activity}")
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
        # colomns = pima[0:1]
        # data = pima[1:]
        # col_names=np.array(colomns).flatten()

        dataset = pd.read_csv(f"{directory}/DAY_{day + 1}.csv", header=None, names=col_names)
        print(dataset)
        # dataset.drop(0, inplace=True)
        # dataset.drop(1, inplace=True)
        print(col_names)
        X = dataset.drop('activity1', axis=1)
        X = dataset.drop('activity2', axis=1)
        y = dataset[activity]
        yArray = np.array(y)
        xArray = np.array(X)
        xArray = xArray.astype(np.float)
        yArray = yArray.astype(np.float)
        # print("========================================")
        # print(xArray)
        # print(yArray)
        X_train, X_test, y_train, y_test = train_test_split(xArray, yArray, test_size=0.20)
        # for x in y:
        #     if type(x) != int:
        #         y[i] = int(x)
        #     i=i+1
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print("Precision = {}".format(precision_score(y_test, y_pred, average='macro')))
        print("Recall = {}".format(recall_score(y_test, y_pred, average='macro')))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        resualt = f"==========\n{directory}/DAY_{day + 1} \n Precision: {format(precision_score(y_test, y_pred, average='macro'))}" \
                  f"\n Accuracy: {accuracy_score(y_test, y_pred)}" \
                  f"\n Recall: {format(recall_score(y_test, y_pred, average='macro'))} \n"
        resultFile.write(resualt)
    resultFile.close()
