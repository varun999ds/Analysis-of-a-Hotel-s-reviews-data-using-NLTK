# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:30:43 2019

@author: desktop
"""

#importing the datasets
import pandas as pd
import numpy as np

#Reading the dataset with pandas
data = pd.read_csv('hotel_reviews.csv', sep='delimiter', header=None, engine='python')
data.rename(columns = {0: 'Reviews'}, inplace = True)

from textblob import TextBlob
def sentiment_textblob(feedback):
    senti = TextBlob(feedback)
    polarity = senti.sentiment.polarity 
    if  polarity < 0:
        label = 0 
    else:
        label = 1
    return (label)
data["Sentiments"] = data["Reviews"].apply(lambda x: sentiment_textblob(x))

#preprocessing of the dataset 
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return(text)

data["review_clean"] = data["Reviews"].apply(lambda x: clean_text(x))


#giving sentimental tokens for all the features using vader sentimentintensityanalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
data["sentiments"] = data["Reviews"].apply(lambda x: sid.polarity_scores(x))
data = pd.concat([data.drop(['sentiments'], axis=1), data['sentiments'].apply(pd.Series)], axis=1)

# add number of characters column
data["nb_chars"] = data["Reviews"].apply(lambda x: len(x))

# add number of words column
data["nb_words"] = data["Reviews"].apply(lambda x: len(x.split(" ")))

'''
# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = data["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
data = pd.concat([data, doc2vec_df], axis=1)
'''

# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = None)
tfidf_result = tfidf.fit_transform(data["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = data.index
data = pd.concat([data, tfidf_df], axis=1)

# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data["review_clean"])

# highest positive sentiment reviews (with more than 5 words)
positive_reviews = data[data["nb_chars"] >= 25].sort_values("pos", ascending = False)[["Reviews", "pos"]]

# lowest negative sentiment reviews (with more than 5 words)
negative_reviews = data[data["nb_words"] >= 25].sort_values("neg", ascending = False)[["Reviews", "neg"]]

import seaborn as sns

for x in [0, 1]:
    subset = data[data['Sentiments'] == x]
    
    # Draw the density plot
    if x == 0:
        label = "bad reviews"
    else:
        label = "good reviews"
    sns.distplot(subset['compound'], hist = False, label = label)
   
sns.countplot(x='Sentiments', data=data)


# feature selection
label = "Sentiments"
ignore_cols = [label,"Reviews", "review_clean"]
features = [c for c in data.columns if c not in ignore_cols]

# split the data into train and test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data[label], test_size = 0.20, random_state = 42)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)

from sklearn import metrics
# Predicting the Test set results
y_pred = rf.predict(X_test)
print('Random_Forest_classifier %s' % metrics.accuracy_score(y_test, y_pred))

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data[features])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(data[features])



# Visualising the clusters
plt.scatter(data[features][y_kmeans == 0, :], data[features][y_kmeans == 0, :], s = 100, c = 'red', label = 'Negative')
plt.scatter(data[features][y_kmeans == 1, :], data[features][y_kmeans == 1, :], s = 100, c = 'green', label = 'Positive')
plt.title('Sentimental analysis')
plt.xlabel('Reviews')
plt.ylabel('sentiments')
plt.legend()
plt.show()










