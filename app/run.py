import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('dataclean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names =  list(genre_counts.index)
    columns_counts = df.shape[0]
    column_names = df.columns
   
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
               

                )
            ],

            'layout': {
                'title': 'Distribution of message by genre ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
             
                Bar(
                    x=column_names,
                    y=columns_counts,
                
                    #orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of Columns',
                'yaxis': {
                    'title': "Columns"
                },
                'xaxis': {
                    'title': "Counts",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]

    
    

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    # render web page with plotly graphs

    return render_template('master.html', ids=ids, graphJSON=graphJSON )

# web page that handles user query and displays model results
@app.route('/go')
def go():
    columns_counts = df.shape[0]
    column_names = df.columns
    # save user input in query
    query = request.args.get('query', '') 
    graphs = [
        {
            'data': [
                Bar(
                    x=column_names,
                    y=columns_counts
                )
            ],

            'layout': {
                'title': 'Disaster Categories',
                'yaxis': {
                    'title': "Count"
                },
             
            }
        }
     
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
         ids=ids, graphJSON=graphJSON,
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()