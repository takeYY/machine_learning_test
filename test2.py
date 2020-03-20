import numpy as np
import pandas as pd
import sklearn

house_price = pd.read_csv("./houseprice_text.csv")
# SalePrice: 予測する物件価格, OverallQual: 物件の品質, GrLivArea: １F以上の間取りの広さ, TotRmsAbvGrd: １F以上の部屋数
print(house_price.head())

y = house_price["SalePrice"]
X = house_price.drop(["Id","SalePrice"],axis=1)

y_array = np.array(y)
X_array = np.array(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.4, random_state=0)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0)

print(rfr.fit(X_train,y_train))

y_pred = rfr.predict(X_test)

from sklearn.metrics import mean_squared_error

print("予測精度：",np.sqrt(mean_squared_error(y_pred, y_test)))


### ここから演習問題
house_price2 = pd.read_csv("./houseprice_assignment_question.csv")

y2_train = y_array
x2_train = X_array

rfr2 = RandomForestRegressor(random_state=0)

print(rfr2.fit(x2_train,y2_train))

X2 = house_price2.drop("Id",axis=1)
X2_array = np.array(X2)

#y2_pred = rfr2.predict(X2_array)
y2_pred = rfr.predict(X2_array)

print(y2_pred)

df = pd.DataFrame({"Id":house_price2["Id"],"SalePrice":y2_pred})

print(df.head())

df.to_csv("predict_SalePrice.csv",index=False)