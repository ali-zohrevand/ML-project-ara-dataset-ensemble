from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
def lr(x_test, x_train, y_train, y_test, directory, day: int):
    resultFile = open(f"lr-{directory}-{day}-result.txt", "a")
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    y_pred = logisticRegr.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    Accuracy = accuracy_score(y_test, y_pred)
    resultFile.write(format(confusion_matrix(y_test, y_pred)))
    resultFile.write(format(classification_report(y_test, y_pred)))
    return y_pred, precision, recall, Accuracy
