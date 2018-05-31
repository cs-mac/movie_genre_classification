from bs4 import BeautifulSoup as soup
import requests
import PTN
import os
from zipfile import ZipFile
from difflib import SequenceMatcher
import shutil
from os import popen

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class Scrapper:
    def __init__(self, link):
        page_html = requests.get(link).content
        self.soup = soup(page_html, 'html.parser')

    def get_id(self, id):
        return soup.find_all(id=id)


class SubSceneScrapper(Scrapper):
    # subscene constants
    SUBSCENE_DOMAIN = 'https://subscene.com'
    QUERY_URI = '/subtitles/title?q='

    # TODO : add ability to search for series
    def __init__(self, movie_name, is_filename=False):
        """
        Used to initialized scrapper
        :param movie_name: name of the file or the direct name of the media  
        :param is_filename: indicated 
        """
        if is_filename:
            self.filename = movie_name
            parsed_info = PTN.parse(movie_name)
            movie_name = parsed_info['title']

        Scrapper.__init__(self, SubSceneScrapper.SUBSCENE_DOMAIN +
                          SubSceneScrapper.QUERY_URI +
                          movie_name.replace(' ', '+'))

        self.movie_name = movie_name

    def __search_media(self):
        """
        Scrapes subscene for movie search 
        :return: A formatted dict representing search results 
        """
        search_result_div = self.soup.find_all('div', 'search-result')[0].contents
        current_category = ''
        movie_results = {
            'Exact': [],
            'Popular': [],
            'Close': [],
            'TV-Series': [], #added this as some where under category TV-series whilst still being a movie
        }
        for tag in search_result_div:
            if tag.name == 'h2':
                current_category = tag.get_text()
            elif tag.name == 'ul':
                for list_item in tag.contents:
                    if list_item.name == 'li':
                        movie_results[current_category].append({
                            'uri': list_item.div.a['href'],
                            'text': list_item.div.a.get_text()
                        })
        return movie_results

    @staticmethod
    def __get_subtitles_from_uri(uri):
        """
        Queries for subtitles with a uri of the movie found by search_media()
        :param uri: uri of the movie subtitles 
        :return: a formatted dict of the subtitles with links and languages
        """
        scrapper = Scrapper(SubSceneScrapper.SUBSCENE_DOMAIN + uri)
        #results_table_contents = scrapper.soup.find_all('tbody')[0].children

        results_table_contents = scrapper.soup.find_all("tr")
       
        subtitles = {}

        for item in results_table_contents:
            if item.name == 'tr':
                if item.td['class'] == ['a1']:
                    rating = None
                    classes = item.td.a.span['class']
                    if 'positive-icon' in classes:
                        rating = 'good'
                    elif 'neutral-icon' in classes:
                        rating = 'neutral'
                    elif 'bad-icon' in classes:
                        rating = 'bad'
                    language = item.td.a.span.get_text().strip(' \r\n\t')
                    subtitle = {
                        'uri': item.td.a['href'],
                        'title': item.td.a.span.next_element.next_element.next_element.get_text().strip(' \r\n\t'),
                        'rating': rating,
                    }

                    if language in subtitles:
                        subtitles[language].append(subtitle)
                    else:
                        subtitles[language] = [subtitle]

        return subtitles

    def get_subtitles(self, must_be_exact=False):
        """
        :return: subtitles of all languages found for the given movie 
        """
        search_result = self.__search_media()
        if not search_result['Exact']:
            if must_be_exact:
                raise ValueError("Couldn't find an exact match for '{}'".format(self.movie_name))
            elif search_result['Popular']:
                return self.__get_subtitles_from_uri(search_result['Popular'][0]['uri'])
            elif search_result['Close']:
                return self.__get_subtitles_from_uri(search_result['Close'][0]['uri'])
            else:
                raise ValueError("Couldn't find a match for '{}'".format(self.movie_name))
        else:
            return self.__get_subtitles_from_uri(search_result['Exact'][0]['uri'])

    def get_best_match_subtitle(self, language):
        """
        returns the subtitle that best matches the subtitle filename
        :param language: the language to be searched in
        :return: a subtitle dict
        """
        subtitles = self.get_subtitles()
        max_similarity = 0
        best_match = None
        for subtitle in subtitles[language]:
            similarity = similar(subtitle['title'], self.filename)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = subtitle

        return best_match

    def download_subtitle_to_path(self, subtitle, path):
        """
        Downloads subtitle file to a given path
        :param subtitle: a subtitle dict
        :param path: the directory for the subtitle to be downloaded in
        """
        # getting subtitle download page html
        subtitle_download_page_html = requests.get(SubSceneScrapper.__get_subtitle_full_link(subtitle)).content

        # scraping download uri from html
        download_uri = soup(subtitle_download_page_html, 'html.parser').find_all(id="downloadButton")[0]['href']

        # add subscene domain to uri
        full_download_link = SubSceneScrapper.SUBSCENE_DOMAIN + download_uri

        # downloading subtitle zip file
        downloaded_zip = requests.get(full_download_link)

        # saving zipfile
        zip_file_name = subtitle['title'] + ".zip"
        with open(zip_file_name, 'wb') as outfile:
            outfile.write(downloaded_zip.content)

        # extracting zipfile
        zip_object = ZipFile(zip_file_name, 'r')
        zip_object.extractall('temp/')
        zip_object.close()

        # removing extracted zipfile
        os.remove(zip_file_name)

        # get subtitle file path in temp folder
        subtitle_file_path = SubSceneScrapper.__get_subtitle_from_temp()

        # move the file to the specified path
        shutil.move(subtitle_file_path, path + self.filename + '.srt')

        # remove temp folder
        shutil.rmtree('temp')

    @staticmethod
    def __get_subtitle_full_link(subtitle):
        return SubSceneScrapper.SUBSCENE_DOMAIN + subtitle['uri']

    @staticmethod
    def __get_subtitle_from_temp():
        for _, _, filenames in os.walk("temp/"):
            for filename in filenames:
                # if is hidden file ignore
                if filename[0] != '.':
                    return 'temp/' + filename





