#!/usr/bin/python3

import sys
import os
import re
import itertools
import collections
import numpy as np
import progressbar #to prevent confusion this needs progressbar2 (pip3 install progressbar2 --user)
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from sklearn.svm import SVC
from movie_classification import uniques, remove_markup, remove_hearing_impaired, remove_speaker, parse_subtitle, tokenize, to_list
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def doc2vec_model(genres):
    '''
    Creates a doc2vec model, and saves it
    '''
    features = []
    all_genres = []
    nltk_stopword_set = set(stopwords.words("english")) #179 words
    scikit_stopword_set = set(stop_words.ENGLISH_STOP_WORDS) #318 words
    union_stopword_set = nltk_stopword_set | scikit_stopword_set # 378 words

    for genre in genres:
        filenames = [files for files in os.listdir("subtitles/" + genre)]
        file_counter = 0
        for file in filenames:
            if file_counter == 400: #max amount of files per genre
                break
            file_counter += 1
            data = parse_subtitle(genre, file)
            try:
                len(data[0][1]) #check if file uses correct time format (e.g. 12:12:12)
            except IndexError:
                file_counter -= 1
                continue
            dialogue = [remove_speaker(remove_hearing_impaired(remove_markup(item[3]))) for item in data if item[5] >= 3]
            dialogue_one_list = list(itertools.chain.from_iterable([tokenize(line) for line in dialogue]))
            bag = uniques([tok if not tok.isupper() else tok.lower() for tok in dialogue_one_list], union_stopword_set) 
            features.append(bag)
            all_genres.append(genre)

    features = [to_list(str(lst)) for lst in features]
    tagged_data = [TaggedDocument(words=bow, tags=[str(idx)]) for idx, bow in enumerate(features)]

    max_epochs = 200
    vec_size = 20 
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.025,
                    min_count=1,
                    dm = 0) #dm = 0 uses bow, dm = 1 preserves word order
      
    model.build_vocab(tagged_data)

    print("#### TRAINING DOC2VEC MODEL\n")

    bar = progressbar.ProgressBar(maxval=max_epochs).start()
    for idx, epoch in enumerate(range(max_epochs)):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
        bar.update(idx)

    bar.finish()

    model.save("d2v_400.model")
    print("Model Saved")

def main():
    categories = ["Comedy", "Drama", "Documentary", "Horror"]
    doc2vec_model(categories)

if __name__ == '__main__':
    main()


