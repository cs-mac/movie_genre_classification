#!/usr/bin/env python3
#Using a sligtly altered version of: https://github.com/mmghannam/subscenery to scrape the subtitles

import sys
import pandas as pd
from scrapper import SubSceneScrapper #needs pip3 install parse-torrent-name --user
import time
import progressbar #needs pip3 install progressbar2 --user
import os
from zipfile import BadZipFile

def clean_directory(directory):
    '''
    Removes files from directory that are created by the scrapper and fail to get removed
    '''
    directory_content = os.listdir(directory)
    for item in directory_content:
        if item.endswith(".zip"):
            os.remove(os.path.join(directory, item))

def get_subs(movie):   
    '''
    Get subtitles for a movie and puts it to the specified path
    '''
    scrapper = SubSceneScrapper(movie.title, is_filename=True)
    scrapper.get_subtitles()
    best_match = scrapper.get_best_match_subtitle('English')
    scrapper.download_subtitle_to_path(best_match, 'subtitles/'+movie.genres+'/')


def filter_no_subs(file, movies_with_no_subs):
    '''
    If we can not find subtitles of a movie -> remove that movie from our data, and creates a csv file that contains
    all movie subtitles downloaded
    '''
    final_csv_file = open("final_movies.csv",'w')
    final_csv_file.write("movieId,title,year,genres\n")

    for row in file.itertuples(index=False):
        if "," in row.title and row.movieId not in movies_with_no_subs:
            title = '"' + row.title + '"'
            final_csv_file.write(str(row.movieId)+","+title+","row.year","+row.genres+"\n")
        elif row.movieId not in movies_with_no_subs:
            final_csv_file.write(str(row.movieId)+","+row.title+","row.year","+row.genres+"\n")    

    final_csv_file.close()


def main(argv):
    file = "movies2_transformed.csv" #data from the movielens 20m dataset, with around 27000 different movies

    print("Searching for subtitles, please be patient...\n")

    header = ["movieId", "title", "year", "genres"]
    movie_genres = pd.read_csv(file, sep = ",", names=header, skiprows=1) 

    no_subs = []

    bar = progressbar.ProgressBar(maxval=len(movie_genres)).start()

    for idx, movie in enumerate(movie_genres.itertuples(index=False)):
        try:
           get_subs(movie)
           time.sleep(2)
        except:
           no_subs.append(movie.movieId)
           time.sleep(2)
           pass
        bar.update(idx)

    bar.finish()      

    print("\nFiltering movies with no subtitles from our data, please be patient...")
    filter_no_subs(movie_genres, no_subs)

    clean_directory(os.getcwd())
    print("Finished!")

if __name__ == '__main__':
    main(sys.argv)