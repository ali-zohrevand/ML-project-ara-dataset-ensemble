from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# //https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
def knn(x_test, x_train, y_train, y_test, directory, day: int):
    resultFile = open(f"knn-{directory}-{day}-result.txt", "a")
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    Accuracy = accuracy_score(y_test, y_pred)
    resultFile.write(format(confusion_matrix(y_test, y_pred)))
    resultFile.write(format(classification_report(y_test, y_pred)))
    return y_pred, precision, recall, Accuracy
