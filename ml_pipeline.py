import numpy as np
import pandas as pd

training_data=pd.read_csv('storepurchasedata.csv')

training_data.describe()

x=training_data.iloc[:,:-1].values #converts to numpy & omits last column
y=training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #we scale age & salary
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""# Build a classification model
### We are using KNN Classifier in this exple
*n_neighbors = 5 -* Number of neighbors
*metric = 'minkowski' , p = 2* - For Euclidean distance calculation
"""

from sklearn.neighbors import KNeighborsClassifier
#minkowski is for eucledian distance
classifier = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski', p=2)

#Model Training
classifier.fit(x_train, y_train) #we train with data, since model is known

y_pred = classifier.predict(x_test) #prediction to compare with y_test
y_prob = classifier.predict_proba(x_test)[:,1]#prediction probability>0.5 gut

"""#To test for model accuracy use confusion matrix
bsp:     test    predict   value_for_matrix
 model|   1        1        True positive
          1        0        False Negative
          0        1        False positive
          0        0        True  negative
           value= True positive + True  negative
                  ______________________________  (geteilt durch)
          True positive + True  negative +  False Negative +   False positive       
"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score #model accuracy

print(accuracy_score(y_test, y_pred))

#for better understanding a report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

#ensure data is numpy and scaled
new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))

new_pred_proba=classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]

#Picking the model & Standard Scaler

import pickle  # to serialise in byte 

model_file = "classifier.pickle"

pickle.dump(classifier, open(model_file,'wb')) #wb- file is open for writing & in binary mode

scaler_file="sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))