import sys
import numpy as np
import pandas as pd
import sqlalchemy
import string
import nltk

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages', engine).set_index('id')
    X = df['message']
    genre_dummies = pd.get_dummies(df['genre'], drop_first=True)
    dummy_cols = genre_dummies.columns
    X = pd.concat([X, genre_dummies], axis=1)
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    categories = Y.columns
    print(X.head())
    return X, Y, categories, dummy_cols


def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    return list(map(lambda token: lemmatizer.lemmatize(token).lower().strip(), tokens))


# TODO: It seems like ColumnTransformer should be able to accomplish this, but I wasn't
# able to get that to work.
class CustomCountVectorizer:
    """A simple wrapper around CountVectorizer to deal with only the 'message' column of X"""
    def __init__(self, *args):
        self.count_vectorizer = CountVectorizer(tokenizer=tokenize, *args)

    def set_params(self, **params):
        """Sets params in the underlying CountVectorizer"""
        self.count_vectorizer.set_params(**params)
        return self

    def fit(self, X, y=None):
        """Fits X['message'] in the underlying CountVectorizer"""
        self.count_vectorizer.fit(X['message'], y)
        return self

    def transform(self, X):
        """transforms X['message'] in the underlying CountVectorizer"""
        result = self.count_vectorizer.transform(X['message'])
        print(result)
        return result


def build_model(dummy_cols):
    """
    This builds a model that will take the text, tokenize it and transform using count->tfidf transformers,
    then combine them with the genre data, to pass into a random forest classifier.
    :param dummy_cols: columns holding the genre dummy vars
    :return:
    """
    text_pipeline = Pipeline([
        ('vect', CustomCountVectorizer()),
        ('tfidf', TfidfTransformer())
    ])
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('select_dummies', ColumnTransformer(transformers=[('columns', 'passthrough', dummy_cols)], remainder='drop')),
            ('text_pipeline', text_pipeline)
        ])),
        ('clf', RandomForestClassifier())
    ])

    # Note: This severely curtails the number of parameters in the interests of running the model in a reasonable time.
    # However, you may uncomment these (and add more values), but expect extreme delays.
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 2), (2, 2)),
        # 'features__text_pipeline__vect__stop_words': ('english', None),
        # 'features__text_pipeline__vect__max_df': (0.5, 1.0),
        'features__text_pipeline__vect__max_features': [5000],
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
        # 'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    f1_scores = []
    for i, category in enumerate(category_names):
        y_test = Y_test.values[:,i]
        y_pred = Y_pred[:, i]
        print(np.unique(y_test))
        print(np.unique(y_pred))
        f = f1_score(y_test, y_pred, zero_division=0)
        f1_scores.append(f)
        print("Classification report for {}:".format(category))
        print(classification_report(y_test, y_pred, zero_division=0))
    print("Average f1 score: ", np.mean(f1_scores))


def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names, dummy_cols = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # ct = make_column_transformer(('passthrough', make_column_selector(['message'])))
        # ct = ColumnTransformer([('m', 'passthrough', cble)])
        # print(ct.fit_transform(X_train))
        # return

        print('Building model...')
        model = build_model(dummy_cols)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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