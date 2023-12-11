# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:54:16 2023

@author: kubra
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

classification_df=pd.read_csv('data_preprocessing/cl_dataset.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


from sklearn.metrics import accuracy_score


x=classification_df.drop(['Title'],axis=1)
y=classification_df['Title']


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)



# Create the kNN classification model
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Make predictions on training and test data
y_train_pred = knn_classifier.predict(X_train)
y_test_pred = knn_classifier.predict(X_test)

# Evaluating training and test accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("kNN Training Accuracy:", train_accuracy)
print("kNN Test Accuracy:", test_accuracy)





# Create the AdaBoost classification model
ada_classifier = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada_classifier.fit(X_train, y_train)

# Make predictions on training and test data
y_train_pred = ada_classifier.predict(X_train)
y_test_pred = ada_classifier.predict(X_test)

# Evaluate training and test accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("AdaBoost Training Accuracy:", train_accuracy)
print("AdaBoost Test Accuracy:", test_accuracy)



#Random Forest Model 

# Create the Random Forest classification model
start_time = time.time()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
end_time = time.time()

train_time = end_time - start_time

start_time = time.time()
# Make predictions on training and test data
y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

# Evaluating training and test accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
end_time = time.time()
test_time = end_time - start_time
print("Random Forest Training Accuracy:", train_accuracy)
print("Random Forest Test Accuracy:", test_accuracy)      




# Adaboost sınıflandırıcı modelini oluşturma
base_model = DecisionTreeClassifier()  # Baz model olarak Decision Tree kullanıldı
model = AdaBoostClassifier(base_model)

# Hiperparametre arama uzayı
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'base_estimator__max_depth': [3, 5, 7]
}

# Grid Search ile hiperparametre optimizasyonu
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# En iyi parametreler ve skorlar
print("Adaboost Best Parameters:", grid_search.best_params_)
print("Adaboost Best Training Accurancy:", grid_search.best_score_)




def graph(column_name):
    

    
    # Grafik oluşturma
    fig, ax = plt.subplots()
    
    # Sütunu grafik olarak çiz
    ax.plot(classification_df[column_name])
    
    # Grafik ayarları
    ax.set_xlabel('Data Points')
    ax.set_ylabel(f'{column_name}')
    ax.set_title(f'{column_name} Graph')
    
    # Grafik gösterme
    plt.show()

graph("Comment Count")
graph("retweet Count")
graph("View count")


correlation_matrix = x.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Özellik Korelasyon Matrisi')
plt.show()


# Sınıf etiketlerine göre veri noktalarını görselleştirme
def data_visualization(x_label,y_label):
    
    plt.scatter(x[x_label], x[y_label])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Data Points Distribution')
    plt.colorbar()
    plt.show()
    
data_visualization('Like Count','View count')
data_visualization('Comment Count','View count')
data_visualization('retweet Count','View count')
data_visualization('Like Count','View count')
data_visualization('countOfPositive','countOfNegative')


#comparing the training results
#comparing the test results

score1, score2, score3 = 1.0, 0.8291457286432161,  0.6733668341708543
scoreo1, scoreo2, scoreo3 = 0.9534883720930233, 0.8488372093023255, 0.3953488372093023
scoresnon = [score1, score2, score3]
scores = [scoreo1, scoreo2, scoreo3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Training Results': scores, 'Test results': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['Random Forest', 'AdaBoost', 'kNN'])

ax.set_title("Training and Test Results")
ax.legend(['Train','Test'])
plt.show()



# Confusion matrix oluşturma
cm = confusion_matrix(y_test, y_test_pred)

# Confusion matrix'in görselleştirilmesi
classes = np.unique(y_test)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='Gerçek Etiket',
       xlabel='Tahmin Edilen Etiket')

# Confusion matrix'in hücreleri için metin yazma
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.show()

with open('classification_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)
    

# Eğitim ve test süresini hesaplama
print("Train Time:", train_time, "second")
print("Test Time:", test_time, "second")