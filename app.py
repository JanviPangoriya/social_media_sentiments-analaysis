from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from nltk import PorterStemmer
from sklearn.linear_model import PassiveAggressiveClassifier
import re

filename = "model.pkl"
clf = pickle.load(open(filename,"rb"))
cv = pickle.load(open("transform.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get("message", False)
        message =message.replace("@[\w]*","")
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
