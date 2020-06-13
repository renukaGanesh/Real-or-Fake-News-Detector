import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("news.csv")
data.columns

#eda
#label encoding the label column

sns.countplot('label',data=data)
data['label'].value_counts()

# Data to plot
labels = 'Real','Fake'
sizes = [3171,3164]
colors = ['yellowgreen', 'lightcoral']
explode = (0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False)

plt.axis('equal')
plt.show()

    #DataFlair - Get the labels
    labels=data.label
    labels.head()
    
    
from nltk.corpus import stopwords
from wordcloud import WordCloud  

wc = WordCloud (max_words = 30000, stopwords = set(stopwords.words("english"))).generate(" ".join(data.title))
plt.imshow(wc)
    
    
    
    
x_train,x_test,y_train,y_test=train_test_split(data['text'], labels,test_size=0.2,random_state=7)
help(TfidfVectorizer)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7) #stop words and also the terms with more than 0.7 fr are also discarded
#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
tfidf_train.get_feature_names())

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#Predict on the test set and calculate accuracy
y_pred=pac.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
