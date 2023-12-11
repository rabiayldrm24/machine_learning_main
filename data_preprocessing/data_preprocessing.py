import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#project

dataset= pd.read_excel("all_tweets.xlsx") 
#print("number of rows in the original dataset",len(dataset))
#print(dataset.head())
#print(dataset.shape)
#print(dataset.columns)
#print(dataset.describe())
#print(dataset.info())
#print(dataset.corr())

#dataset.hist()
#print(dataset["Title"].unique())



dataset.drop_duplicates(inplace=True)

#pd.read_excel("all_tweets.xlsx") 

like_column = 'Like Count'
comment_column = 'Comment Count'
retweet_column = 'retweet Count'
view_column = 'View count'

dataset[like_column]

def convert_k_m(column_name):
    for i in range(len(dataset)):
        deger = str(dataset.at[i, column_name])
        if 'K' in deger:
            deger = deger.replace('K', '')
            deger = float(deger) * 1000
            dataset.at[i, column_name] = int(deger)
        elif  'M' in deger:
            deger = deger.replace('M', '')
            deger = float(deger) * 1000000
            dataset.at[i, column_name] = int(deger)
        
convert_k_m(like_column)
convert_k_m(comment_column)
convert_k_m(retweet_column)
convert_k_m(view_column)

dataset[comment_column].fillna(dataset[comment_column].mean(), inplace=True)
dataset[comment_column] = dataset[comment_column].astype(int)
dataset[like_column].fillna(dataset[like_column].mean(), inplace=True)
dataset[like_column] = dataset[like_column].astype(int)
dataset[view_column].fillna(dataset[view_column].mean(), inplace=True)
dataset[view_column] = dataset[view_column].astype(int)
dataset[retweet_column].fillna(dataset[retweet_column].mean(), inplace=True)
dataset[retweet_column] = dataset[retweet_column].astype(int)


text_column = dataset["Text"]
#kelime sayısını ayırma
def countOfWord(text):
    liste = []
    for x in range(0,len(text)):
        liste.append(len(text[x].split()))
    return liste  

countOfText = countOfWord(text_column)
dataset["countOfWords"] = countOfText

data_positive = pd.read_csv("PositiveWordsEng.csv")
data_negative = pd.read_csv("NegativeWordsEng.csv")
data_positive = data_positive["PositiveWords"]
data_negative = data_negative["NegativeWords"]

def search_tw(data_words,data_tw):
    liste = []
    for tweet in data_tw:
        count = 0 
        tweet = str(tweet).lower().split()
        for word in data_words:
            if word in tweet:
                count+=1
        liste.append(count)
    return liste 

countOfPositive = search_tw(data_positive,text_column)
countOfNegative = search_tw(data_negative,text_column)


dataset["countOfPositive"] = countOfPositive

dataset["countOfNegative"] = countOfNegative

dataset.to_csv("dataset_c.csv",index=False)

#print(dataset.columns)

#########################


#for regression

one_hot_df = pd.get_dummies(dataset['Title'])
dataset['ID'] = range(1, len(dataset) + 1)
one_hot_df['ID'] = range(1, len(one_hot_df) + 1)


one_hot_df[one_hot_df == True] = "1"
one_hot_df[one_hot_df == False] = "0"

merged_df = pd.merge(dataset, one_hot_df, on='ID')

merged_df = merged_df.drop('ID',axis=1)
merged_df = merged_df.drop('Title',axis=1)                              
merged_df.to_csv("dataset_reg.csv",index=False)

#print(dataset.columns)



#Concanate text-mining dataset and drop the user-name, date-of-tweet in data.csv


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import text
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('dataset_reg.csv')

# Drop unnecessary columns
data = data.drop(columns=['Id', "User Name","Date of Tweet"])

# Convert text features to lowercase
data['Text'] = data['Text'].str.lower()

# Separate features and target variable
X = data.drop(['Like Count'], axis=1)
y = data['Like Count']

my_stop_words = list(text.ENGLISH_STOP_WORDS)

#numeric featrue names
coulmn_names = list(X.columns)
coulmn_names.remove("Text")


# Preprocess text data
Tfidf_transformer = TfidfVectorizer(stop_words=my_stop_words)
count_transformer = CountVectorizer(stop_words=my_stop_words)
scaler = StandardScaler()

# Combine text and numeric features
preprocessor = ColumnTransformer(transformers=[
    ("T_Text", Tfidf_transformer, "Text"),
   
    ('C_Text', count_transformer, 'Text'),
    ('numeric', 'passthrough', coulmn_names)
])
# Fit and transform the preprocessor to the data
X_preprocessed = preprocessor.fit_transform(X)
X_preprocessed_df = pd.DataFrame(X_preprocessed.toarray(), columns=preprocessor.get_feature_names_out())
X_preprocessed_df.to_csv('tweets_in.csv')  
y.to_csv('tweets_out.csv')  

#Feature importance with orange 
data_y = pd.read_csv("tweets_out.csv")
#print(data_y.columns)

data_x = pd.read_csv("tweets_in.csv")

#r2

#print(data_x["C_Text__mysteries"].unique()) 0,1
#print(data_x["C_Text__replying"].unique()) 0,1
#print(data_x["T_Text__replying"].unique()) 0,1
#print(data_x["T_Text__manchester"].unique()) diff. values

#mse

#first four is same
#print(data_x["C_Text__manchester"].unique())
#print(data_x["T_Text__believe"].unique())
#print(data_x["T_Text__mark"].unique())

columns = ["C_Text__mysteries","C_Text__replying","T_Text__replying","T_Text__manchester","C_Text__manchester","T_Text__believe","T_Text__mark"]

text_df = data_x[columns]
text_df.index.name ="Id"
text_df.to_csv("text_df.csv",index=True)


#dataset for regression
text_df = pd.read_csv("text_df.csv")
dataset = pd.read_csv("dataset_reg.csv")
dataset = dataset.drop("Text",axis=1)
dataset = dataset.drop("User Name",axis=1)
dataset = dataset.drop("Date of Tweet",axis=1)

dataset.dropna(inplace=True)
#print("Number of rows in the cleaned dataset:",len(dataset))
#print(dataset.info())


dataset[comment_column] = dataset[comment_column].astype(int)
dataset[like_column] = dataset[like_column].astype(int)
dataset[view_column] = dataset[view_column].astype(int)
dataset[retweet_column] = dataset[retweet_column].astype(int)


dataset.to_csv("dataset_reg_noText.csv",index=False)
dataset_noText = dataset = pd.read_csv("dataset_reg_noText.csv")


merged_df = pd.merge(dataset_noText, text_df, on='Id')
merged_df.to_csv("reg_dataset.csv",index=False)

data_new = pd.read_csv("reg_dataset.csv")

#dataset for cluster and classification
dataset = pd.read_csv("dataset_c.csv")
dataset = dataset.drop("Text",axis=1)
dataset = dataset.drop("User Name",axis=1)
dataset = dataset.drop("Date of Tweet",axis=1)

dataset.to_csv("dataset_c_noText.csv",index=False)
dataset_noText = dataset = pd.read_csv("dataset_c_noText.csv")


merged_df = pd.merge(dataset_noText, text_df, on='Id')
merged_df.to_csv("cl_dataset.csv",index=False)

data_new = pd.read_csv("cl_dataset.csv")

