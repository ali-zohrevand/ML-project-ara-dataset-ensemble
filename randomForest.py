from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def random_forst(x_test, x_train, y_train, y_test, directory, day: int):
    resultFile = open(f"randomForest-{directory}-{day}-result.txt", "a")
    classifier = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets
    classifier.fit(x_train, y_train)
    # predictin on the test set
    y_pred = classifier.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    Accuracy = accuracy_score(y_test, y_pred)
    resultFile.write(format(confusion_matrix(y_test, y_pred)))
    resultFile.write(format(classification_report(y_test, y_pred)))
    return y_pred, precision, recall, Accuracy
