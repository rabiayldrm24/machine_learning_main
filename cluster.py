#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:54:30 2023

@author: esmanur
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####################################################
#Data read 
data = pd.read_csv("data_preprocessing/reg_dataset.csv")

X = data.drop(["Like Count"], axis=1)
y = data["Like Count"]

data.info()


data["LikeCount"] = data["Like Count"]
data["CommentCount"] = data["Comment Count"]
data["ViewCount"] = data["View count"]
####################################################

sns.kdeplot(data["LikeCount"]) #yoğunluk
plt.show()

sns.distplot(data['LikeCount'], bins=30, kde=False)
plt.show()

sns.jointplot(x ='CommentCount', y='LikeCount', data=data[data['LikeCount'] > 1000],kind='hex', 
              gridsize=20)
plt.show()

sns.violinplot(
    x='ViewCount',
    y='LikeCount',
    data=data[data.ViewCount.isin(data.ViewCount.value_counts()[:5].index)]
)
plt.show()


plt.figure(figsize=(10,10))
sns.boxplot(x="CommentCount", y="LikeCount",  data=data.iloc[:200])
plt.xticks(rotation=90)

sns.jointplot(data=X, x='Comment Count', y='View count', kind="reg", color="#ce1414")


for feature in X.columns:
    plt.hist(data[feature], bins=20)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    
# İki özelliğin ilişkisini gösteren scatter plot
plt.scatter(data["Like Count"], data['Comment Count'])
plt.ylabel("Like Count")
plt.xlabel('Comment Count')
plt.show()   

#####################################################
#Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)


#Train kısmında outlierları çıkarıp modeli eğiteceğim sonra test kısmında outlier çıkarımı yapmadan modeli eğiteceğim
#Ardından en iyi farkı veren ile outlier çıkarımı modelini kaydedeceğim


#####################################################
#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_scaled,y, test_size=0.30, random_state=42)


######################################################
#DBSCN Algoritması

from sklearn.cluster import DBSCAN


dbscn = DBSCAN(eps=2.462, min_samples=2,metric="euclidean") 
clusters_dbscn = dbscn.fit_predict(X_train)


outliers_dbscn = np.where(clusters_dbscn == -1)

plt.scatter(X_train[:, 2], X_train[:, 1], c="blue", alpha=0.5)
# Aykırı verileri ayrıca görselleştirme
plt.scatter(X_train[outliers_dbscn, 2], X_train[outliers_dbscn, 1], c='red', marker='x', alpha=1)
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.title("DBSCN")
plt.show()


print("Number of outliers:", len(outliers_dbscn[0]))
print(outliers_dbscn)

#data_dbscn_outlier = X_train.drop(outliers_dbscn[0],axis = 0)

data_dbscn_outlier = np.delete(X_train,outliers_dbscn[0],axis=0)

######################################################
from sklearn.cluster import KMeans

# K-means kümeleme
kmeans = KMeans(n_clusters=2, random_state=20, n_init="auto")
#kmeans.fit(X)

clusters_kmean = kmeans.fit(X_train)

# Aykırı değerleri tespit etme
distances = clusters_kmean.transform(X_train)
std_dev = np.std(distances, axis=0)
mean_dist = np.mean(distances, axis=0)

threshold = mean_dist + 2 * std_dev #Normal Distribution
outliers_kmean = np.where(np.any(distances > threshold, axis=1))[0]

plt.scatter(X_train[:, 2], X_train[:, 1], c="blue", alpha=0.5)
# Aykırı verileri ayrıca görselleştirme
plt.scatter(X_train[outliers_kmean, 2], X_train[outliers_kmean, 1], c='red', marker='x', alpha=1)
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.title("KMeans")
plt.show()



print("Number of outliers:", len(outliers_kmean))
print(outliers_kmean)

#data_kmean_outlier = data.drop(outliers_kmean,axis = 0)

data_kmean_outlier = np.delete(X_train,outliers_kmean,axis=0)

   
for feature in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(X[feature], y, c=kmeans.labels_, cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel('Lİke count')
    plt.title(f'{feature}')
    plt.show()    
    
correlation_matrix = data.corr()


# Korelasyon matrisini görselleştirme
sns.heatmap(correlation_matrix)
plt.show()    



######################################################
#GMM Cluster

from sklearn import metrics
from sklearn.mixture import GaussianMixture

parameters=['full','tied','diag','spherical']
n_clusters=np.arange(1,21)
results_=pd.DataFrame(columns=['Covariance Type','Number of Cluster','Silhouette Score','Davies Bouldin Score'])
for i in parameters:
    for j in n_clusters:
        gmm_cluster=GaussianMixture(n_components=j,covariance_type=i,random_state=123)
        clusters=gmm_cluster.fit_predict(df_scaled)
        if len(np.unique(clusters))>=2:
           results_=results_.append({
           "Covariance Type":i,'Number ofCluster':j,
           "Silhouette Score":metrics.silhouette_score(df_scaled,clusters),
           'Davies Bouldin Score':metrics.davies_bouldin_score(df_scaled,clusters)}
           ,ignore_index=True)



gmm = GaussianMixture(n_components=2,covariance_type="spherical",random_state=123)
clusters_gmm = gmm.fit(X_train)

distances = clusters_gmm.score_samples(X_train)
std_dev = np.std(distances, axis=0)
mean_dist = np.mean(distances, axis=0)

threshold = mean_dist + 2 * std_dev #Normal Distribution

#thresholz Z-score a göre yapılmıştır
outliers_gmm = np.where(threshold > 3)[0]

plt.scatter(X_train[:, 2], X_train[:, 1], c="blue", alpha=0.5)
plt.scatter(X_train[outliers_gmm, 2], X_train[outliers_gmm, 1], c='red', marker='x')
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.title("GaussianMixture")
plt.show()

print("Aykırı veri sayısı:", len(outliers_gmm))
print(outliers_gmm)

#data_gmm_outlier = data.drop(outliers_gmm,axis = 0)


data_gmm_outlier = np.delete(X_train,outliers_gmm,axis=0)
    
    
##########################################################
#Outlieri hesaplanmış datasetleri fit edip dorğuluk oranını hesaplama
dbsnc_cluster = dbscn.fit_predict(data_dbscn_outlier)
kmeans_cluster = kmeans.fit_predict(data_kmean_outlier)
gmm_cluster = gmm.fit_predict(data_gmm_outlier)

##########################################################
#Outlieri hesaplanmamış datasetleri fit edip dorğuluk oranını hesaplama
dbsnc_cluster_nonoutlier = dbscn.fit_predict(X_test)
kmeans_cluster_nonoutlier = kmeans.fit_predict(X_test)
gmm_cluster_nonoutlier = gmm.fit_predict(X_test)



############################################################
#algoritma karşılaştırma
print("Scores non-outlier\n")

from sklearn.metrics import silhouette_score

score_dbscn = silhouette_score(data_dbscn_outlier, dbsnc_cluster)
score_kmeans = silhouette_score(data_kmean_outlier, kmeans_cluster)
score_gmm = silhouette_score(data_gmm_outlier, gmm_cluster)

print("Silhouette score dbscn nonoutlier: " ,score_dbscn)
print("Silhouette score kmean nonoutlier: " ,score_kmeans)
print("Silhouette score gmm nonoutlier:", score_gmm)



from sklearn.metrics import davies_bouldin_score  #ideal olan 0

dav_dbscn = davies_bouldin_score(data_dbscn_outlier,dbsnc_cluster)
dav_kmean = davies_bouldin_score(data_kmean_outlier,kmeans_cluster)
dav_gmm = davies_bouldin_score(data_gmm_outlier,gmm_cluster)

print("Davies score k mean nonoutlier: ", dav_kmean)
print("Davies score dbscn nonoutlier: ", dav_dbscn)
print("Davies score gmm nonoutlier : ",dav_gmm)
print("\n\n")
############################################################
print("Scores outlier\n")
score_dbscn_outlier = silhouette_score(X_test, dbsnc_cluster_nonoutlier)
score_kmeans_outlier = silhouette_score(X_test, kmeans_cluster_nonoutlier)
score_gmm_outlier = silhouette_score(X_test, gmm_cluster_nonoutlier)

print("Silhouette score dbscn outlier" ,score_dbscn_outlier)
print("Silhouette score kmean outlier: " ,score_kmeans_outlier)
print("Silhouette score gmm outlier:", score_gmm_outlier)


dav_dbscn_outlier = davies_bouldin_score(X_test,dbsnc_cluster_nonoutlier)
dav_kmean_outlier = davies_bouldin_score(X_test,kmeans_cluster_nonoutlier)
dav_gmm_outlier = davies_bouldin_score(X_test,gmm_cluster_nonoutlier)

print("Davies score k mean outlier: ", dav_kmean_outlier)
print("Davies score dbscn outlier: ", dav_dbscn_outlier)
print("Davies score gmm outlier: ",dav_gmm_outlier)


##############################################################
# Silhouette skoru değişimleri
silhouette_scores = [score_dbscn, score_kmeans, score_gmm]
silhouette_scores_outlier = [score_dbscn_outlier, score_kmeans_outlier, score_gmm_outlier]

# Davies-Bouldin skoru değişimleri
davies_scores = [dav_dbscn, dav_kmean, dav_gmm]
davies_scores_outlier = [dav_dbscn_outlier, dav_kmean_outlier, dav_gmm_outlier]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Silhouette skoru değişimleri
plt.subplot(2, 1, 1)
plt.plot(silhouette_scores, label="Scores non-outlier")
plt.plot(silhouette_scores_outlier, label='Scores outlier')
plt.xticks(range(len(silhouette_scores)), ['DBSCAN', 'KMeans', 'GMM'])
plt.ylabel('Silhouette skoru')
plt.legend()

# Davies-Bouldin skoru değişimleri
plt.subplot(2, 1, 2)
plt.plot(davies_scores, label='Scores non-outlier')
plt.plot(davies_scores_outlier, label='Scores outlier')
plt.xticks(range(len(davies_scores)), ['DBSCAN', 'KMeans', 'GMM'])
plt.ylabel('Davies-Bouldin skoru')
plt.legend()

# Grafiği göster
plt.show()

##############################################################


score1, score2, score3 = 0.48572030811677946, 0.38050738642379256, 0.39658785730975155
scoreo1, scoreo2, scoreo3 = 0.4323568378275871, 0.3865528998888909, 0.7704501116325587
scoresnon = [score1, score2, score3]
scores = [scoreo1, scoreo2, scoreo3]

# Değerleri DataFrame'e dönüştürme
df = pd.DataFrame({'Silhouette Scores non-outlier': scores, 'Scores outlier': scoresnon})

# Tabloyu çizdirme
ax = df.plot(kind='bar', figsize=(8, 6), rot=0)
ax.set_xlabel("Models")
plt.xticks(range(len(scores)), ['DBSCAN', 'KMeans', 'GMM'])

ax.set_title("Changes in Silhouette Scores Values")
ax.legend(['Scores outlier','Scores non-outlier'])
plt.show()

############################################################
#Model saving 
import pickle

#Save the trained model as pickle string to disk for future use
filename = "dbscn_model"
pickle.dump(dbscn, open(filename, 'wb'))
