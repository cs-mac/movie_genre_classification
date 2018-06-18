# README #

### Bachelor ###
Name: Chi Sam Mac

Student-number: s2588382

### FILES AT START ###


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


### STEPS ###

Step1: python3 transform_data.py #To get the data of which movies to get subtitles from

Step2: python3 get_subs.py #This downloads all the subtitles for the movies chosen before

Step3: python3 doc2vec.py #Create a doc2vec model 

Step4: python3 movie_classification.py (Don't need grid_search.py/gs_feature_weights.py, already used for parameters/weights in classifier in movie_classification.py)


