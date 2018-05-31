#!/usr/bin/env python3

import sys
import pandas as pd
from collections import Counter

def transform_data(file, genres_chosen):
    '''
    Takes movielens dataset csv file and filter all movies with more than 1 genre and saves the remainder to a new csv file
    '''
    header = ["movieId", "title", "genres"]
    movie_genres = pd.read_csv(file, sep = ",", names=header, skiprows=1) 

    transformed_csv_file = open(file.split(".")[0]+"_transformed.csv",'w')
    transformed_csv_file.write("movieId,title,year,genres\n")

    for row in movie_genres.itertuples(index=False):
        genre_list = row.genres.split('|')
        if len(genre_list) == 1 and row.genres in genres_chosen:
            year = str(row.title.split()[-1])
            title = " ".join(row.title.replace(":","").split()[:-1])
            if "," in title:
                title = '"' + title + '"'
                transformed_csv_file.write(str(row.movieId)+","+title+","+year+","+row.genres+"\n")
            else:
                transformed_csv_file.write(str(row.movieId)+","+title+","+year+","+row.genres+"\n")

    transformed_csv_file.close()

    print("Transformed data to only keep movies with 1 genre, out of the chosen genres!\n")

    return file.split(".")[0]+"_transformed.csv"


def balance_data(file, genres_chosen, max_movies):
    '''
    Takes movielens dataset csv file and balances between movie genres, within a certain range, and saves the balanced 
    data to a new csv file
    '''
    header = ["movieId", "title", "year", "genres"]
    movie_genres = pd.read_csv(file, sep = ",", names=header, skiprows=1) 

    balanced_csv_file = open(file.split(".")[0]+"_balanced.csv",'w')
    balanced_csv_file.write("movieId,title,year,genres\n")

    action_counter = comedy_counter = documentary_counter = drama_counter = \
    horror_counter = thriller_counter = musical_counter = 0

    counters = [action_counter, comedy_counter, documentary_counter, drama_counter, 
    horror_counter, thriller_counter, musical_counter]

    for row in movie_genres.itertuples(index=False):
        for genre, number in zip(genres_chosen, range(7)):
            if row.genres == genre and counters[number] < max_movies:
                counters[number] += 1
                if "," in row.title:
                    title = '"' + row.title + '"'
                    balanced_csv_file.write(str(row.movieId)+","+title+","+row.year+","+row.genres+"\n")
                else:
                    balanced_csv_file.write(str(row.movieId)+","+row.title+","+row.year+","+row.genres+"\n")

    balanced_csv_file.close()

    print("Balanced data better, according to a max movies per genre count!\n")

    return file.split(".")[0]+"_balanced.csv"


def data_info(file, genres_chosen):
    '''
    Takes a movielens dataset csv file and shows some information about the data it contains
    '''
    header = ["movieId", "title", "year", "genres"]
    movie_genres = pd.read_csv(file, sep = ",", names=header, skiprows=1) 

    all_genres_set = set()

    movies_w_1_genre = []

    for row in movie_genres.itertuples(index=False):
        try:
            genre_list = row.genres.split('|')
            if len(genre_list) == 1 and row.genres in genres_chosen:
                movies_w_1_genre.append(*genre_list)
            for genre in genre_list:
                all_genres_set.add(genre)
        except AttributeError:
            movies_w_1_genre.append(row.genres)

    print("\nInfo from file: {}\n".format(file))

    print("Amount of genres: {}\n\nAll genres:\n{}\n".format(len(all_genres_set),all_genres_set)) #all different genres (we can now see we need to remove movies without a listed genre)

    print("Genres chosen: {}\n".format(genres_chosen))

    print("Amount of movies w/ 1 genre: {} out of ({})\n".format(len(movies_w_1_genre),len(list(movie_genres.itertuples())))) #movies with 1 genre with the genre we want

    print(Counter(movies_w_1_genre))


def main(argv):
    #file = "movies.csv" #data from the movielens 100k dataset, with around 9000 different movies
    #file = "movies2.csv" #data from the movielens 20m dataset, with around 27000 different movies

    genres_chosen = ["Action", "Comedy", "Documentary", "Drama", "Horror", "Thriller"]

    #filtered_file = transform_data(file, genres_chosen) #file with movies with only 1 genre
    #balanced_file = balance_data(filtered_file, genres_chosen, 200) #filtering later
    #data_info(filtered_file, genres_chosen)
    data_info("final_movies.csv", genres_chosen)

if __name__ == '__main__':
    main(sys.argv)