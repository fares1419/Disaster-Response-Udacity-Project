import pickle

import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import nltk
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df =  pd.read_sql_table("dataclean", con=engine)
    X = df['message']
    Y = df.iloc[:,( df.columns !='id') & (df.columns!='message')& (df.columns!='original')&       (df.columns!='genre')]
    return X,Y
 
def tokenize(text):
    words=word_tokenize(text)
    Lemmatizer=WordNetLemmatizer()
    
    clean=[]
    
    for word in words:
        word=Lemmatizer.lemmatize(word).lower().strip()
        clean.append(word)
    return clean

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize,lowercase=False)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
        'clf__estimator__n_estimators' : [95, 100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv

    return pipeline
def evaluate_model(model, X_test,Y_test):
   Y_pred=model.predict(X_test)
   for i, column in enumerate(Y_test):
        print("column name:",column)
        print("\n")
        print(classification_report(Y_test[column], Y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()