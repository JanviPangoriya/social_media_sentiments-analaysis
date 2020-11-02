from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk import PorterStemmer
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv("Dataset/training_data.csv")
    def remove_handle(tweet):
        match = re.findall("@[\w]*",tweet)
        for i in match:
            tweet = re.sub(i,'',tweet)
        return tweet
    vector = np.vectorize(remove_handle)
    data['tweets without handle'] = vector(data['tweet'])
    data['tweets without handle'] = data['tweets without handle'].str.replace("[^a-zA-Z#]"," ")
    #tokenize the words
    tokenized_tweets = data['tweets without handle'].apply(lambda x: x.split())
    #sTEMMING THE WORDS
    ps = PorterStemmer()
    tokenized_tweets = tokenized_tweets.apply(lambda x : [ps.stem(word) for word in x])
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
    data['tweets without handle'] = tokenized_tweets
    vectorizer_bow = CountVectorizer(max_features=6000,stop_words='english',ngram_range=(1,4))
    x_bow = vectorizer_bow.fit_transform(data['tweets without handle']).toarray()
    y_bow = data['label']
    from sklearn.model_selection import train_test_split
    xtrain_bow,xtest_bow,ytrain_bow,ytest_bow = train_test_split(x_bow,y_bow,test_size=0.2,random_state=3)
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(xtrain_bow,ytrain_bow)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer_bow.transform(data).toarray()
        my_prediction = nb.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)