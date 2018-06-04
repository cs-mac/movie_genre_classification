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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

def get_high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    '''
    Gets the high information words using chi square measure
    '''
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1
    
    n_xx = label_word_fd.N()
    high_info_words = set()
    
    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)
        
        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        
        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
    
    return high_info_words

def high_information_words(X, y):
    '''
    Get and display info on high info words
    '''
    print("\n#### OBTAINING HIGH INFO WORDS...")

    labelled_words = []
    amount_words = 0
    distinct_words = set()
    for words, genre in zip(X, y):
        labelled_words.append((genre, words))
        amount_words += len(words)
        for word in words:
            distinct_words.add(word)

    high_info_words = set(get_high_information_words(labelled_words, BigramAssocMeasures.chi_sq, 5)) #5 seems best with this amount of data

    print("\tNumber of words in the data: %i" % amount_words)
    print("\tNumber of distinct words in the data: %i" % len(distinct_words))
    print("\tNumber of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words

def wpm(files_used, genres, show_plots=False):
    '''
    Calculate the word/minute of each genre
    '''
    count_movie = 0
    sp = plt.figure(figsize=(16, 9), dpi=150)

    time_features = []
    print("\n#### CALCULATING WORDS PER MINUTE...")
    for genre in genres:
        cnt = 0
        calc_sum = 0
        indices = []
        wpm_list = []
        for file in files_used[genre]:
            subs = parse_subtitle(genre, file)
            cnt += 1
            count_movie += 1
            length_movie_minute = 60*int(subs[-1][1].split(":")[0]) + int(subs[-1][1].split(":")[1]) #time of latest dialogue of a movie
            if length_movie_minute <= 0:
                time_features.append(0)
                continue
            word_freq = 0
            for sub in subs:
                word_freq += len(str(sub[3]).split(" "))
            wpm = word_freq/length_movie_minute #amount of words divived by movie time in minutes 
            wpm_list.append(wpm)
            indices.append(count_movie)
            calc_sum += wpm
            time_features.append(wpm)
        plt.plot(indices, wpm_list, 'o', label=genre)
        plt.axis([0, count_movie+10, 0, 200])
        if cnt > 0:
            print("\t", genre, calc_sum/cnt)

    plt.xlabel('Movies')
    plt.ylabel('Word/minute')
    plt.legend(loc='upper right', numpoints = 1)
    plt.title("Scatterplot WPM")
    if show_plots:
        plt.show()  
    sp.savefig('WPM_scatterplot' + '.png', figsize=(16, 9), dpi=150)
    plt.close()

    return time_features

def dpm(files_used, genres, show_plots=False):
    '''
    Calculate the dialogue/minute of each genre
    '''
    print("\n#### CALCULATING DIALOGUE PER MINUTE...")
    count_movie = 0
    sp = plt.figure(figsize=(16, 9), dpi=150)

    time_features = []

    for genre in genres:
        cnt = 0
        calc_sum = 0
        indices = []
        dpm_list = []
        for file in files_used[genre]:
            subs = parse_subtitle(genre, file)
            cnt += 1
            count_movie += 1
            length_movie_minute = 60*int(subs[-1][1].split(":")[0]) + int(subs[-1][1].split(":")[1])
            if length_movie_minute <= 0:
                time_features.append(0)
                continue
            dpm = len(subs)/length_movie_minute
            indices.append(count_movie)
            dpm_list.append(dpm)
            calc_sum += dpm
            time_features.append(dpm)
        plt.plot(indices, dpm_list,'o', label=genre)
        plt.axis([0, count_movie+10, 0, 40])
        if cnt > 0:
            print("\t", genre, calc_sum/cnt)

    plt.xlabel('Movies')
    plt.ylabel('Dialog/minute')
    plt.legend(loc='upper right', numpoints = 1)
    plt.title("Scatterplot DPM")
    if show_plots:
        plt.show()  
    sp.savefig('DPM_scatterplot' + '.png', figsize=(16, 9), dpi=150)
    plt.close()

    return time_features

def dialogue_distribution(files_used, genres, time_boundry_min=10):
    '''
    Calculate the dialogue distribution of each movie of each genre
    '''
    print("\n#### CALCULATING DIALOGUE DISTRIBUTION OF MOVIES...")
    time_features = []
    for genre in genres:
        for file in files_used[genre]:
            dd_list = []
            subs = parse_subtitle(genre, file)
            length_movie_minute = 60*int(subs[-1][1].split(":")[0]) + int(subs[-1][1].split(":")[1])
            if length_movie_minute <= 0:
                time_features.append([0])
                continue
            check = 0
            dialogue_during_boundry = 0
            for dialogue in subs:
                time_seconds = dialogue[5]
                if time_seconds-check <= time_boundry_min*60:
                    dialogue_during_boundry += len(dialogue[3])
                    continue
                dd_list.append(dialogue_during_boundry)
                dialogue_during_boundry = 0
                check += time_boundry_min*60
            if dialogue_during_boundry > 100:
                dd_list.append(dialogue_during_boundry) #append last part left if size is reasonable
            time_features.append(dd_list)
 
    longest = max(map(len, time_features))
    for lst in time_features:
        if len(lst) < longest:
            lst.extend([0 for _ in range(longest-len(lst))])

    print("\tDone calculating")

    return time_features

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

        #print (metrics.classification_report(y_test, pred, digits=3))

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

def to_list(string):
    '''
    Makes the format that I use before tfidf vectorizing from ['[text]',...] to ['text', ...]
    '''
    string = string[1:-1]
    return [token[1:-1] for token in string.split(', ')]

# def tag(tokens):
#     '''
#     Return a pos tag for each token in a document
#     '''
#     doc = nlp(tokens)
#     return [t.pos_ for t in doc]

# class PosFeatures(TransformerMixin): 
#     """ using POS tags from Spacy """
#     def __init__(self):
#         nlp = spacy.load('en')
        
#     def _tag(tokens):
#         doc = nlp(tokens)
#         return [t.pos_ for t in doc]
        
#     def transform(self, X):
#         return [_tag(word) for word in X]

#     def fit(self, x, y=None):
#         return self

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, subs):
        features = {}
        #features['text'] = [' '.join(to_list(item[0])) for item in subs] #cleaner looking, but same functionality
        features['text'] = [item[0] for item in subs]
        features['text_high'] = [item[1] for item in subs]
        features['wpm'] = [[float(item[2])] for item in subs]
        features['dpm'] = [[float(item[3])] for item in subs]
        features['dd'] = [item[4] for item in subs]
        features['d2v'] = [item[5] for item in subs]
        #features['pos'] = [" ".join(tag(str(sentence))) for sentence in [' '.join(to_list(item[0])) for item in subs]]

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

    #read categories from arguments. e.g. "python3 test.py Drama Comedy Horror"
    categories = []
    for arg in sys.argv[1:]:
        categories.append(arg)

    X, y, files_used = read_files(categories)

    try:
        high_info_words = high_information_words(X, y)

        X_high_info = []
        for bag in X:
            new_bag = []
            for words in bag:
                if words in high_info_words:
                    new_bag.append(words)
            X_high_info.append(new_bag)
    except ZeroDivisionError:
        print("Not enough information too get high information words, please try again with more files.", file=sys.stderr)
        X_high_info = X

    X_wpm = wpm(files_used, categories, show_plots)
    X_dpm = dpm(files_used, categories, show_plots)
    X_dd = dialogue_distribution(files_used, categories)

    doc2vec_model = Doc2Vec.load("d2v_150.model")

    #Reason I don't infer the vector is that I used the data already while training the vector model (with tagged docoments), so I can just retrieve the data
    X_d2v = [doc2vec_model.docvecs[str(i)] for i in range(len(X))]
    #X_d2v = [doc2vec_model.infer_vector(to_list(str(i))) for i in X] 

    X = [(str(x), str(x_high), wpm, dpm, dd, d2v) for x, x_high, wpm, dpm, dd, d2v in zip(X, X_high_info, X_wpm, X_dpm, X_dd, X_d2v)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    clfs = [
        LogisticRegression(solver='lbfgs'), #2nd best close to svm, but not yet used grid search on svm
        SVC(C=10, cache_size=500, class_weight=None, coef0=0.0, #use cv grid search to parameter tune (works best for now with random parameters)
        decision_function_shape=None, degree=3, gamma=0.0001, kernel='linear',
        max_iter=100000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False),
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MultinomialNB(alpha=1.0),
        GradientBoostingClassifier(), #really slow
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
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, binary=True, norm='l2', ngram_range=(1,3))),
                    ('chi-square', SelectKBest(chi2, 300)),
                ])),

                #Pipeline for high info words bag-of-words model 
                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_high')),
                    ('tfidf', TfidfVectorizer(sublinear_tf=True, norm='l2')),
                ])),

                #Pipeline for wpm feature
                ('wpm', Pipeline([
                    ('selector', ItemSelector(key='wpm')),
                    ('scaler', MinMaxScaler()),
                ])),

                #Pipeline for dpm feature
                ('dpm', Pipeline([
                    ('selector', ItemSelector(key='dpm')),
                    ('scaler', MinMaxScaler()),
                ])),

                #Pipeline for dd feature
                ('dd', Pipeline([
                    ('selector', ItemSelector(key='dd')),
                    ('scaler', MinMaxScaler()),
                ])),

                #Pipeline for d2v feature
                ('d2v', Pipeline([
                    ('selector', ItemSelector(key='d2v')),
                    ('scaler', MinMaxScaler()),
                ])),

                #Pipeline for POS tag features
                # ('pos', Pipeline([
                #     ('selector', ItemSelector(key='pos')),
                #     ('words', TfidfVectorizer(sublinear_tf=True, binary=True, norm='l2', ngram_range=(1,3)))
                # ])),

            ],

            # weight components in FeatureUnion #think about using gridsearch on transformer weights
            transformer_weights={ #best = wpm=0.2, dpm=0.2, dd=0, d2v=0.2, pos=0, text=1
                'wpm': .2,
                'dpm': .2,
                'dd': 0,
                'd2v': .2,
                #'pos': 0,
                'text': 1,
                'text_high': 1,
            },
        )),

        # Use a classifier on the combined features
        ('classifier', clfs[1]),
    ])

    train(pipeline, X_train, y_train, categories, show_plots)

    final_pred = pipeline.predict(X_test)
    print("\nScores on test set:\n")
    print(metrics.classification_report(y_test, final_pred, digits=3))

if __name__ == '__main__':
    main()