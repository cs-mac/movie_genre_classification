# README #

### Bachelor ###
Name: Chi Sam Mac

Student-number: s2588382

### How to run the classifier ###
First you need to have all subtitles files (directory subtitle, with subdirectories the genres: Comedy, Drama, Documentary, Thriller, etc.)

The Doc2vec model: d2v_150.model

The movie_classification.py file.

Then run movie_classification.py* as: __python3 movie_classification.py Comedy Drama Documentary Horror__

*POS tag feature commented to increase efficiency of program, as the feature does not add anything to performance of the classifier. To turn it on uncomment the following lines: 35, 392-412, 427, 534-537, 549. 

### Files neccesary to gather all data from scratch ###


Directory "subtitles" with in it directories of the genres ("Action", "Comedy", "Documentary", "Drama", "Horror", "Thriller")


movies.csv (movieLens data roughly 9k different movies)

movies2.csv (movieLens data roughly 27k different movies)


transform_data.py

get_subs.py

scrapper.py

doc2vec.py

grid_search.py (for parameters algorithms)

gs_feature_weights.py (for weights of features)

base_program.py (classifier with basic BOW model)

movie_classification.py


### Steps to gather all data from scratch and run the classifier ###

Step1: python3 transform_data.py #To get the data of which movies to get subtitles from

Step2: python3 get_subs.py #This downloads all the subtitles for the movies chosen before

Step3: python3 doc2vec.py #Create a doc2vec model 

Step4: python3 movie_classification.py (Don't need grid_search.py/gs_feature_weights.py, already used for parameters/weights in classifier in movie_classification.py)


