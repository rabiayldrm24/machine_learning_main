#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:39:08 2023

@author: esmanur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("indoor_data_HAACS.xlsx")
#Dataset
data.columns = data.columns.str.strip()
X = data.drop(['X (Numeric)', 'Y (Numeric)', 'Floor (Categoric)'], axis=1)  

data["XNumeric"] = data["X (Numeric)"]
data["YNumeric"] = data["Y (Numeric)"]
data["Floor"] = data["Floor (Categoric)"]

print(data.columns)
missing_values = data.isnull().sum().sum()

# Print the count of missing values for each column
print(missing_values)
print("number of rows in the original dataset",len(data))
print(data.head())
print(data.shape)
print(data.columns)
print(data.describe())
print(data.info())
print(data.corr())


sns.kdeplot(data["F_dB_std"]) #yoğunluk
plt.show()

sns.distplot(data["F_dB_mean"], bins=30, kde=False)
plt.show()


sns.violinplot(
    x='F_dB_std',
    y='YNumeric',
    data=data[data.YNumeric.isin(data.YNumeric.value_counts()[:].index)]
)
plt.show()


plt.figure(figsize=(10,10))
sns.boxplot(x="F_dB_std", y="Y (Numeric)",  data=data.iloc[:200])
plt.xticks(rotation=90)



for feature in X.columns:
    plt.hist(data[feature], bins=20)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    
# İki özelliğin ilişkisini gösteren scatter plot
plt.scatter(data["YNumeric"], data['F_dB_std'])
plt.ylabel("YNumeric")
plt.xlabel('F_dB_std')
plt.show()   

#X numeric için model eğitimi


X = data.drop(['X (Numeric)', 'Y (Numeric)', 'Floor (Categoric)'], axis=1)  # Input features
X_numeric = data['X (Numeric)']  
Y_numeric = data['Y (Numeric)']  
floor = data['Floor (Categoric)']

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X,X_numeric, test_size=0.3, random_state=100)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(X_train, y_train)

y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
# Model performansını değerlendirin
mse_rf_test = mean_squared_error(y_test, y_pred_test)
r2_rf_test = r2_score(y_test, y_pred_test)
print('Test Mean Squared Error Random Forest X_numeric:', mse_rf_test)
print('Test R-squared Score Random Forest X_numeric:', r2_rf_test)

mse_rf_train = mean_squared_error(y_train, y_pred_train)
r2_rf_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_rf_train)
print('Train R-squared Score Random Forest X_numeric:', r2_rf_train)

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
# Train the model
linear.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear.predict(X_test)

# Evaluate the model
mse_linear_test = mean_squared_error(y_test, y_pred)
r2_linear_test = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error Linear X_numeric:', mse_linear_test)
print('R-squared Lİnear X_numeric:', r2_linear_test)

mse_linear_train = mean_squared_error(y_train, y_pred_train)
r2_linear_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_linear_train)
print('Train R-squared Score Random Forest X_numeric:', r2_linear_train)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse_knn_test = mean_squared_error(y_test, y_pred)
r2_knn_test = r2_score(y_test, y_pred)
print('Mean Squared Error knn: X_numeric', mse_knn_test)
print('R-squared Score knn: X_numeric', r2_knn_test)

mse_knn_train = mean_squared_error(y_train, y_pred_train)
r2_knn_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_knn_train)
print('Train R-squared Score Random Forest X_numeric:', r2_knn_train)

train1, train2, train3 = r2_rf_train, r2_linear_train, r2_knn_train
test1, test2, test3 = r2_rf_test, r2_linear_test, r2_knn_test
scoresnon = [train1, train2, train3]
scores = [test1, test2, test3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Test': scores, 'Train': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['RF', 'Linear', 'KNN'])

ax.set_title("Changes in train Scores Values")
ax.legend(['Scores train','Scores test'])
plt.show()


print("####################################")
#Y (Numeric)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X,Y_numeric, test_size=0.3, random_state=100)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Model performansını değerlendirin
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
# Model performansını değerlendirin
mse_rf_test = mean_squared_error(y_test, y_pred_test)
mse_rf_test = r2_score(y_test, y_pred_test)
print('Test Mean Squared Error Random Forest X_numeric:', mse_rf_test)
print('Test R-squared Score Random Forest X_numeric:', mse_rf_test)

mse_rf_train = mean_squared_error(y_train, y_pred_train)
r2_rf_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_rf_train)
print('Train R-squared Score Random Forest X_numeric:', r2_rf_train)

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
# Train the model
linear.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear.predict(X_test)

mse_linear_test = mean_squared_error(y_test, y_pred)
r2_linear_test = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error Linear X_numeric:', mse_linear_test)
print('R-squared Lİnear X_numeric:', r2_linear_test)

mse_linear_train = mean_squared_error(y_train, y_pred_train)
r2_linear_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_linear_train)
print('Train R-squared Score Random Forest X_numeric:', r2_linear_train)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse_knn_test = mean_squared_error(y_test, y_pred)
r2_knn_test = r2_score(y_test, y_pred)
print('Test Mean Squared Error knn: X_numeric', mse_knn_test)
print('Test R-squared Score knn: X_numeric', r2_knn_test)

mse_knn_train = mean_squared_error(y_train, y_pred_train)
r2_knn_train = r2_score(y_train, y_pred_train)
print('Train Mean Squared Error Random Forest X_numeric:', mse_knn_train)
print('Train R-squared Score Random Forest X_numeric:', r2_knn_train)
print("####################################")

train1, train2, train3 = r2_rf_train, r2_linear_train, r2_knn_train
test1, test2, test3 = r2_rf_test, r2_linear_test, r2_knn_test
scoresnon = [train1, train2, train3]
scores = [test1, test2, test3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Test': scores, 'Train': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['RF', 'Linear', 'KNN'])

ax.set_title("Changes in train Scores Values")
ax.legend(['Scores train','Scores test'])
plt.show()


#Floor
X_train, X_test, y_train, y_test = train_test_split(X,floor, test_size=0.3, random_state=100)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)

adaboost.fit(X_train,y_train)

y_pred_test = adaboost.predict(X_test)

accuracy_ada_test = accuracy_score(y_test, y_pred_test)

# Print the accuracy
print('Accuracy ada test:', accuracy_ada_test)

y_pred_train = adaboost.predict(X_train)

accuracy_ada_train = accuracy_score(y_train, y_pred_train)

# Print the accuracy
print('Accuracy ada train:', accuracy_ada_train)




from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = knn_model.predict(X_test)

accuracy_knn_test = accuracy_score(y_test, y_pred_test)

# Print the accuracy
print('Accuracy knn test:', accuracy_knn_test)

y_pred_train = knn_model.predict(X_train)

accuracy_knn_train = accuracy_score(y_train, y_pred_train)

# Print the accuracy
print('Accuracy knn train:', accuracy_knn_train)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

y_pred_test = rf_model.predict(X_test)

accuracy_rf_test = accuracy_score(y_test, y_pred_test)

# Print the accuracy
print('Accuracy rf test:', accuracy_rf_test)

y_pred_train = knn_model.predict(X_train)

accuracy_rf_train = accuracy_score(y_train, y_pred_train)

# Print the accuracy
print('Accuracy rf train:', accuracy_rf_train)


train1, train2, train3 = accuracy_rf_train, accuracy_knn_train, accuracy_ada_train
test1, test2, test3 = accuracy_rf_test, accuracy_knn_test, accuracy_ada_test
scoresnon = [train1, train2, train3]
scores = [test1, test2, test3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Test': scores, 'Train': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['RF', 'Linear', 'KNN'])

ax.set_title("Changes in train Scores Values")
ax.legend(['Scores train','Scores test'])
plt.show()

#########################################

importances = rf_model.feature_importances_

# Create a DataFrame with feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Get the top 10 feature names
top_10_features = importance_df.head(10)['Feature'].tolist()

# Print the top 10 feature names
print("Top 10 features:")
for feature in top_10_features:
    print(feature)