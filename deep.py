import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from boost import xgboost
from randomForest import random_forst
from svm_lib import svm_function
columnsString = "daySection,Ph1,Ph2,Ir1,Fo1,Fo2,Di3,Di4,Ph3,Ph4,Ph5,Ph6,Co1,Co2,Co3,So1,So2,Di1,Di2,Te1,Fo3,activity1,activity2"
col_names = columnsString.split(",")
col_names = columnsString.split(",")
dataset = pd.read_csv(f"data-House1.csv", header=None, names=col_names)
X = dataset.drop('activity1', axis=1)
y = dataset['activity1']
yArray = np.array(y)
xArray = np.array(X)
xArray = xArray.astype(np.float)
yArray = yArray.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(xArray, yArray, test_size=0.20)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units=21,activation='softmax'))
model.add(Dense(units=21,activation='softmax'))
model.add(Dense(units=21,activation='softmax'))
model.add(Dense(units=21,activation='softmax'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile( optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train,
          y=y_train,
          epochs=1000,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
eval=model.evaluate(x=X_test,
          y=y_test)
print(eval)
model_loss = pd.DataFrame(model.history.history)
# model_loss.plot(kind='line')
import matplotlib.pyplot as plt
# plt.savefig('output.png')
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
