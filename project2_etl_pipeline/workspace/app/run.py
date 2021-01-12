import json
import string

import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')


app = Flask(__name__)


def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    s = stopwords.words('english')
    result = []
    for token in clean_tokens:
        if token not in s:
            result.append(token)

    return result


# load data
engine = create_engine('sqlite:///../data/disaster_data.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/model.pickle")


def message_genre_bar_chart(df):
    """
    Creates a dictionary suitable for plotly of a barchart of the genres
    :param df: pandas dataframe containing data
    :return: A dictionary of the barchart info
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    return {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }


def category_bar_chart(df):
    """
    Creates a dictionary suitable for plotly of a barchart of the labeled categories
    :param df: pandas dataframe containing data
    :return: A dictionary of the barchart info
    """
    label_names = df.drop(['message', 'original', 'genre', 'id'], axis=1).columns
    label_counts = []
    for column in label_names:
        label_counts.append(df[column].sum())
    return {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Labelled Categories',
                'yaxis': {
                    'title': "Count",
                    'type': 'log'
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }


def top_words_bar_chart(df, n=10):
    """
    Returns a barchart of the most common word stems in 'message', aside from stopwords
    :param df: Dataframe containing 'message' column
    :param n: number of words to keep, default 10
    :return:
    """
    messages = df['message'].values
    word_counts = {}
    for message in messages:
        tokens = tokenize(message)
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    items = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    items = items[0:n]
    words = list(map(lambda x: x[0], items))
    counts = list(map(lambda x: x[1], items))
    return {
            'data': [
                Bar(
                    x=words,
                    y=counts
                )
            ],

            'layout': {
                'title': 'Most common word stems (outside stopwords)',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = [
        message_genre_bar_chart(df),
        category_bar_chart(df),
        top_words_bar_chart(df)
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    print(df.head())
    # use model to predict classification for query
    # For now, assume that this is a 'direct' genre message
    # TODO: Add a dropdown so that the user can select the genre
    row = [(query, '', 0, 0 )]
    to_predict = pd.DataFrame(row, columns=['message', 'original', 'news', 'social'], index=[0])
    classification_labels = model.predict(to_predict)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()