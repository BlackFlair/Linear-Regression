import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#%matplotlib inline # Magic function that renders the figure in a notebook

companies = pd.read_csv("")

x = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values # 4 : Number of rows to be selected

companies.head() # Display in row and column format

sns.heatmap(companies.corr()) # corr() : coordinates

labelEncoder = LabelEncoder()
x[:, 3] = labelEncoder.fit_transform(x[:,3]) # 3 : Column with string data
oneHotEncoder = OneHotEncoder(catagorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

x = x[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Test Set Result Prediction : ", y_pred)
print("Linear Regression Coefficient : ", regressor.coef_)
print("Linear Regression Intercept : ", regressor.intercept_)

score = r2_score(y_test, y_pred)

if score*100 > 91.0:
    print("Trained Model EFFICIENT with R2 Value : ", score)
else:
    print("Trained Model INEFFICIENT with R2 Value : ", score)