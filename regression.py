# -- coding: utf-8 --
"""
Created on Mon May  1 15:54:16 2023

@author:rabia
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


regression_df=pd.read_csv('data_preprocessing/reg_dataset.csv')

def graph(column_name):
    
    
    # Grafik oluşturma
    fig, ax = plt.subplots()
    
    # Sütunu grafik olarak çizme
    ax.plot(regression_df[column_name])
    
    # Grafik ayarları
    ax.set_xlabel('Data Points')
    ax.set_ylabel(f'{column_name}')
    ax.set_title(f'{column_name} Graph')
    
    # Grafik gösterme
    plt.show()

graph("Comment Count")
graph("retweet Count")


X=regression_df.drop(['Like Count'],axis=1)
y=regression_df['Like Count']

correlation_matrix = X.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Özellik Korelasyon Matrisi')
plt.show()

# Sınıf etiketlerine göre veri noktalarını görselleştirme
def data_visualization(x_label,y_label):
    
    plt.scatter(X[x_label], X[y_label])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Data Points Distribution')
    plt.colorbar()
    plt.show()
    
data_visualization('Comment Count','View count')
data_visualization('retweet Count','View count')
data_visualization('Art','Health')
data_visualization('Politics','Sport')
data_visualization('countOfPositive','countOfNegative')



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=100)

# Random Forest Regressor modelini eğitme
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_train_predictions = rf_model.predict(X_train)
rf_train_mae = mean_absolute_error(y_train, rf_train_predictions)
rf_train_r2 = r2_score(y_train, rf_train_predictions)
rf_test_predictions = rf_model.predict(X_test)
rf_test_mae = mean_absolute_error(y_test, rf_test_predictions)
rf_test_r2 = r2_score(y_test, rf_test_predictions)

# AdaBoost Regressor modelini eğitme
adaboost_model = AdaBoostRegressor()
adaboost_model.fit(X_train, y_train)
adaboost_train_predictions = adaboost_model.predict(X_train)
adaboost_train_mae = mean_absolute_error(y_train, adaboost_train_predictions)
adaboost_train_r2 = r2_score(y_train, adaboost_train_predictions)
adaboost_test_predictions = adaboost_model.predict(X_test)
adaboost_test_mae = mean_absolute_error(y_test, adaboost_test_predictions)
adaboost_test_r2 = r2_score(y_test, adaboost_test_predictions)

# Linear Regression modelini egitme
start_time = time.time()
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_train_predictions = linear_model.predict(X_train)
linear_train_mae = mean_absolute_error(y_train, linear_train_predictions)
linear_train_r2 = r2_score(y_train, linear_train_predictions)
end_time = time.time()
train_time = end_time - start_time

start_time = time.time()
linear_test_predictions = linear_model.predict(X_test)
linear_test_mae = mean_absolute_error(y_test, linear_test_predictions)
linear_test_r2 = r2_score(y_test, linear_test_predictions)
end_time = time.time()
test_time = end_time - start_time

# Sonuçlar
print("Random Forest:")
print("Train MAE:", rf_train_mae)
print("Train R2:", rf_train_r2)
print("Test MAE:", rf_test_mae)
print("Test R2:", rf_test_r2)

print("\nAdaBoost:")
print("Train MAE:", adaboost_train_mae)
print("Train R2:", adaboost_train_r2)
print("Test MAE:", adaboost_test_mae)
print("Test R2:", adaboost_test_r2)

print("\nLinear Regression:")
print("Train MAE:", linear_train_mae)
print("Train R2:", linear_train_r2)
print("Test MAE:", linear_test_mae)
print("Test R2:", linear_test_r2)
print("Train Time:", train_time, "saniye")
print("Test Time:", test_time, "saniye")

#comparing the training results
#comparing the test results

score1, score2, score3 = 0.9663617973384698, 0.9516390879541713,  0.7622865842301308
scoreo1, scoreo2, scoreo3 = 0.6176591444835061, 0.623656236764593, 0.8511556012001812
scoresnon = [score1, score2, score3]
scores = [scoreo1, scoreo2, scoreo3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Training Results': scores, 'Test results': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['Random Forest', 'AdaBoost', 'Lineer Regression'])

ax.set_title("Training and Test Results")
ax.legend(['Train','Test'])
plt.show()

# Modelin tahminlerini ve gerçek değerleri alma
predictions = linear_model.predict(X_test)
residuals = y_test - predictions

# Residual scatter plotunu çizdirme
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Sıfır çizgisi (hataların ortalaması)
plt.xlabel('Tahminler')
plt.ylabel('Hatalar')
plt.title('Residual Scatter Plot')
plt.show()

# Önişleme adımları içeren bir pipeline oluşturma
pipeline = make_pipeline(StandardScaler(), LinearRegression())

# Hiperparametreler ve değer aralıklarını belirleyin
hyperparameters = {
    'linearregression__fit_intercept': [True, False],
    'linearregression__normalize': [True, False]
}

# GridSearchCV ile hiperparametre araması yapma
grid_search = GridSearchCV(pipeline, hyperparameters, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

# En iyi modeli seçme
best_model = grid_search.best_estimator_

# Test veri seti üzerinde tahmin yapma
predictions = best_model.predict(X_test)

# MAE ve R2 değerlerini hesaplama
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Sonuçları yazdırma
print("Optimistic MAE:", mae)
print("Optimistic R2:", r2)

with open ('regression_model.pkl', 'wb') as file:
    pickle.dump(linear_model,file)