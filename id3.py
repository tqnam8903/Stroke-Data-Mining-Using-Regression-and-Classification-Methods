import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv('.\data_daxuly_logistic.csv.csv')
# print(dt)
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']].values
y = data['stroke'].values  # Cột dự đoán là "stroke"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dect = DecisionTreeClassifier(criterion= 'entropy', splitter='best', max_depth=12).fit(X_train, y_train)
#criterion phuong phap do luong chat luong phan chia
y_predict = dect.predict(X_test)
y_Test = np.array(y_test)

print("Accuracy score: %4f" % accuracy_score(y_Test,y_predict))
print("Tỷ lệ dự đoán sai: %4f" % (1 - accuracy_score(y_Test,y_predict)))
print("Precision score: %4f" % precision_score(y_Test,y_predict))
print("Recall score: %4f" %recall_score(y_Test,y_predict))
print("F1 score: %4f" %f1_score(y_Test,y_predict))