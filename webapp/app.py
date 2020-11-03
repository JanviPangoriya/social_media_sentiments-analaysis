import pathlib
from typing import re

import flask
import pickle
import numpy as np

# Use pickle to load in the pre-trained model.

with open(f'webapp/lrVectorizer.pkl', 'rb') as f:
    model = pickle.load(f)

# removing handle names
def remove_handle(tweet):
    match = re.findall("@[\w]*",tweet)
    for i in match:
        tweet = re.sub(i,'',tweet)
    return tweet

app = flask.Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method =='POST':

        text = flask.request.form['text']
        #vector = np.vectorize(remove_handle(text))
        #print(vector)
        new_text = text.split()
        return flask.render_template('main.html', original_input={'text' : text}, result=text)

if __name__ == '__main__':
    app.run()