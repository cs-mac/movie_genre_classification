#!/usr/bin/python3

import sys
import os
import re
import itertools
import collections

import spacy
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

from gensim.models.doc2vec import Doc2Vec

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, StratifiedKFold
from sklearn import metrics

#nlp = spacy.load('en')

def uniques(words, badwords=False):
    '''
    Returns set of unique words, and filters badwords from them
    '''
    if not badwords:
        return set(words)
    return set(words) - set(badwords)

def remove_markup(text):
    '''
    Removes markup like <i></i> and returns the text that is left
    '''
    return re.sub('<[^<]+?>', '', text)

def remove_hearing_impaired(text):
    '''
    Removes hearing impaired information, targeting specfically conventions used in hearing impaired subtitles like:
    (HEARING IMPAIRED PART) or music parts like "♪ Yeah, ah"
    '''
    return re.sub("♪.*♪", "", re.sub("♪.*\n.*", "", re.sub("[\(\[].*?[\)\]]", "$", text)))
    
def remove_speaker(text):
    '''
    Sometimes in subtitles the speaker will be displayed e.g. Speaker1: "hi". This function removes the speaker, and 
    only leaves the dialogue
    '''
    return re.sub(".*:\n.*", "", re.sub(".*: .*\n.*", "$", text))

def parse_subtitle(genre, file):
    '''
    Parses subtitles of a movie into list with tuples cosisting of the conversation Id, 
    start time, end time, content, minute in the movie, second in the movie of a dialogue
    '''
    data = open('subtitles/' + genre + '/' + file, 'r', encoding='UTF-8', errors='ignore')
    data_listed = [list(g) for b,g in itertools.groupby(data, lambda x: bool(x.strip())) if b]

    subs = []
    conversation_id = 0
    for sub in data_listed:
        if len(sub) >= 3: 
            sub = [x.strip() for x in sub]

            conversation_id = sub[0]
            start_end = sub[1]
            dialogue = " ".join(sub[2:])

            if len(start_end.split(' --> ')) == 2:
                start, end = start_end.split(' --> ') 

                if len(start) == 12 and len(end) == 12:
                    try:
                        minute = int(start[:2]) * 60 + int(start[3:5])
                        second = int(start[:2]) * 3600 + int(start[3:5]) * 60 + int(start[6:8])
                    except:
                        minute = 0
                        second = 0
                    subs.append((conversation_id, start, end, dialogue, minute, second))

    return subs

def tokenize(string):
    '''
    Takes a string and tokenizes it, but keeps words containing ' the same (e.g. It'll)
    '''
    words = "".join([c if c.isalnum() or c is "'" else " " for c in string]) 
    words_clean = words.split()
    return words_clean

def read_files(genres):
    '''
    Read in the files of a the genre directories in the subtitle directory, and return bag of words and genres
    '''
    print("#### READING FILES...")
    features = []
    all_genres = []
    nltk_stopword_set = set(stopwords.words('english')) #179 words
    scikit_stopword_set = set(stop_words.ENGLISH_STOP_WORDS) #318 words
    union_stopword_set = nltk_stopword_set | scikit_stopword_set # 378 words
    files_used = collections.defaultdict(list)

    for genre in genres:
        filenames = [files for files in os.listdir('subtitles/' + genre)]
        file_counter = 0
        #word_sum = 0
        for file in filenames:
            if file_counter == 150:
                break
            #if word_sum >= 160000:
            #    break
            file_counter += 1
            #snow = SnowballStemmer('english')
            data = parse_subtitle(genre, file)
            try:
                len(data[0][1]) #check if file uses correct time format (e.g. 12:12:12)
            except IndexError:
                file_counter -= 1
                continue
            #if item[5] >= 3 to remove things like "created by [Someone]" or "Subtitles by [Someone]"
            dialogue = [remove_speaker(remove_hearing_impaired(remove_markup(item[3]))) for item in data if item[5] >= 3]
            dialogue_one_list = list(itertools.chain.from_iterable([tokenize(line) for line in dialogue])) 
            #word_sum += len(dialogue)
            bag = uniques([tok if not tok.isupper() else tok.lower() for tok in dialogue_one_list], union_stopword_set) 
            #bag = uniques([snow.stem(tok) for tok in dialogue_one_list], union_stopword_set) #stemming makes it slower and slightly less accuracte
            #bag = uniques([Word(tok).lemmatize() for tok in dialogue_one_list], union_stopword_set) #lemmatizing made it slightly worse
            features.append(bag)
            all_genres.append(genre)
            files_used[genre].append(file)
        print ("\tGenre %s, %i files read" % (genre, file_counter))

    print("\tTotal, %i files read" % (len(features)))
    return features, all_genres, files_used

def train(pipeline, X, y, categories, show_plots=False):
    '''
    Train the classifier and evaluate the results
    '''
    print("\n#### EVALUATION...")

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)

    print(pipeline.named_steps['classifier'])

    accuracy = 0
    confusion_m = np.zeros(shape=(len(categories),len(categories)))
    kf = StratifiedKFold(n_splits=10).split(X, y)
    pred_overall = np.array([])
    y_test_overall = np.array([])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trained  = pipeline.fit(X_train, y_train) 
        pred = pipeline.predict(X_test)
        accuracy += metrics.accuracy_score(y_test, pred)
        #precision, recall, fscore = precision_recall_fscore_support(y_test, pred, labels=categories, average='macro')[:-1] #average
        precision, recall, fscore = metrics.precision_recall_fscore_support(y_test, pred, average=None, labels=categories)[:-1] #per class
        confusion_m = np.add(confusion_m, metrics.confusion_matrix(y_test, pred, labels=categories))
        
        pred_overall = np.concatenate([pred_overall, pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

        print(metrics.classification_report(y_test, pred, digits=3))

    print("\n"+"Average accuracy: %.6f"%(accuracy/10) + "\n")

    print (metrics.classification_report(y_test_overall, pred_overall, digits=3))
    print('Confusion matrix')
    print(confusion_m)

    plt.figure(figsize = (16, 9), dpi=150)
    sn.set(font_scale=1.4) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 16}) #font size
    hm.set(xticklabels = categories, yticklabels = categories)
    plt.title(str(pipeline.named_steps['classifier']).split("(")[0] + ' Confusion Matrix')
    if show_plots:
        plt.show()

    hm.figure.savefig(str(pipeline.named_steps['classifier']).split("(")[0] + '_confusion_matrix' + '.png', figsize = (16, 9), dpi=150)

    plt.close()

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, subs):
        features = {}
        features['text'] = [item for item in subs]

        return features

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def main():
    show_plots = False #set to True to show plots, False to not show plots

    #read categories from arguments. e.g. "python3 test.py Comedy Drama Documentary Horror"
    categories = []
    for arg in sys.argv[1:]:
        categories.append(arg)

    X, y, files_used = read_files(categories)

    X = [str(x) for x in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    clfs = [
        SVC(C=10, cache_size=500, class_weight=None, coef0=0.0, #parameters found using grid_search.py
        decision_function_shape=None, degree=3, gamma=0.0001, kernel='linear',
        max_iter=100000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False),
    ]

    pipeline = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[
                #Pipeline bag-of-words model 
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, binary=True, norm='l2')),
                ])),
            ],
        )),

        # Use a classifier on the combined features
        ('classifier', clfs[0]),
    ])

    train(pipeline, X_train, y_train, categories, show_plots)

if __name__ == '__main__':
    main()