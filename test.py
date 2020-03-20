
import numpy as np
import pandas as pd
import sklearn

titanic = pd.read_csv("./titanic_text.csv")
#survived: 1=生存, sex: 0=男性, Pclass: チケットのクラス 1=高い, fare: チケットの価格
#print(titanic.head())
y = titanic["Survived"]
X = np.array(titanic.drop(["Survived","PassengerId"],axis=1))
y_array = np.array(y)
X_array = np.array(X)

#print("y_array: ", y_array.shape)
#print("X_array: ", X_array.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.4, random_state=0)

#print("=========================================================")
#print("X_train: ", X_train.shape)
#print("y_train: ", y_train.shape)
#print("X_test: ", X_test.shape)
#print("y_test: ", y_test.shape)
#print("=========================================================")

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print("rfc:",rfc.fit(X_train, y_train))

y_pred = rfc.predict(X_test)

#print("=========================================================")
from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred, y_test))

#print("=========================================================")
#print("X_test:",X_test)



### ここから演習問題
titanic2 = pd.read_csv("./titanic_assignment_question.csv")

print(titanic2.head())

y2_train = y_array
x2_train = X_array
X2 = np.array(titanic2.drop("PassengerId",axis=1))
X2_array = np.array(X2)

print("X2_array:",X2_array.shape)

rfc2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print("rfc2:",rfc2.fit(x2_train,y2_train))

y_pred2 = rfc2.predict(X2_array)
#y_pred2 = rfc.predict(X2_array)

print("y_pred2:",y_pred2)
#print("passengerIds",titanic2["PassengerId"])

df = pd.DataFrame({"PassengerId":titanic2["PassengerId"],"Survived":y_pred2})

print("\n")
#print(df)

df.to_csv("survived_predict.csv",index=False)
