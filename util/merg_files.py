import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from boost import xgboost
def merge_files(directory):
    earlyMorning = 18000
    morning = 43200
    noon = 57600
    night = 72000
    for day in range(30):
        print(f"Day {day+1}")
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
        f = open(f"data-{directory}.csv", "a")
        f.write(output)
        f.close()
