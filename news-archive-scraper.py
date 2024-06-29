#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import datetime as dt
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class Newspaper_Archives_Scraper :
    """ This class represents the scraper, both does the scraping and saves the data to a DataFrame.

    Attributes:
        start_date (datetime): First day of articles scraping
        end_date (datetime): Last day of articles scraping
        archive_urls (dic): date and url of all the archive articles for this date
        article_urls (list of tuples): urls, titles and dates of all the articles

    Méthodes:
        generate_archive_urls(self): Generate all archive URLs between start_date and end_date.
        fetch_article_urls(self): Fetch all article URLs, titles and dates from archive pages in parallel using ThreadPoolExecutor.
        save_to_dataframe(self): Save the collected article data to a DataFrame."""



    def __init__(self, params):
          """Initialize the scraper """

          self.url = params.get('url')
          self.archive_urls = []
          self.article_urls = []
          self.params = params



    def remove_duplicates_by_link(self):
        """
        Supprime les articles en double basés sur les liens.

        Args:
        - articles (list of dict): Liste de dictionnaires avec 'title' et 'link' pour chaque article.

        Returns:
        - list of dict: Liste de dictionnaires sans doublons.
        """
        seen_links = set()
        unique_articles = []

        for article in self.article_urls:
            link = article['link']
            if link not in seen_links:
                seen_links.add(link)
                unique_articles.append(article)

        self.article_urls = unique_articles

    def save_to_dataframe(self):
        """Save the collected article data to a DataFrame."""
        self.df = pd.DataFrame(self.article_urls)
       

    def add_article_url(self, futures) :
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scraping archive pages"):
            result = future.result()
            self.article_urls.extend(result)

    def process(self):
        self.archive_urls = self.generate_archive_urls()
        self.fetch_article_urls()
        self.remove_duplicates_by_link()
        self.save_to_dataframe()



class Archive_date(Newspaper_Archives_Scraper) :

    def __init__(self, params):
        super().__init__(params) # Call the parent class constructor with params
        self.start_date = datetime.strptime(self.params.get('start_date'), "%d.%m.%Y")
        self.end_date = datetime.strptime(self.params.get('end_date'), "%d.%m.%Y")
        self.url_type, self.to_replace = self.url_format()

    def url_format(self):
            regex_patterns = {
              r'(\d{2})-(\d{2})-(\d{4})': 'DD-MM-YYYY',
              r'(\d{2})\.(\d{2})\.(\d{4})': 'DD.MM.YYYY',
              r'(\d{4})-(\d{2})-(\d{2})': 'YYYY-MM-DD',
              r'(\d{4})/(\d{2})/(\d{2})': 'YYYY/MM/DD',
              r'(\d{4})\.(\d{2})\.(\d{2})': 'YYYY.MM.DD',
              r'(\d{4})-(\d{1,2})-(\d{1,2})': 'YYYY-M-D',
              r'(\d{4})/(\d{1,2})/(\d{1,2})': 'YYYY/M/D'
          }

            detected_format = None

            for regex_pattern, format_name in regex_patterns.items():
                if re.search(regex_pattern, self.url):
                    detected_format = format_name
                    to_replace = regex_pattern
                    break

            if detected_format is None:
                raise ValueError(f"Format de date non supporté dans l'URL modèle : {self.url}")

            return detected_format, to_replace


    def generate_archive_urls(self):
      # Définir les formats de conversion
      format_mappings = {
          'DD-MM-YYYY': '%d-%m-%Y',
          'DD.MM.YYYY': '%d.%m.%Y',
          'YYYY-MM-DD': '%Y-%m-%d',
          'YYYY/MM/DD': '%Y/%m/%d',
          'YYYY.MM.DD': '%Y.%m.%d',
          'YYYY-M-D': '%Y-%m-%d',
          'YYYY/M/D': '%Y/%m/%d'
      }

      delta = self.end_date - self.start_date
      archive_urls = {}
      for i in range(delta.days + 1):
          day = self.start_date + timedelta(days=i)


          # Convertir l'objet datetime en chaîne de caractères avec le format détecté
          formatted_date = day.strftime(format_mappings[self.url_type])

          # Remplacer la date dans l'URL format
          #new_url = re.sub(r'\d{4}[-/.]?\d{2}[-/.]?\d{2}', formatted_date, self.url)

          a = re.search(self.to_replace, self.url)
          new_url = self.url.replace(a.group(0), formatted_date )
          archive_urls[formatted_date] = new_url

      return archive_urls

    def fetch_article_urls(self):
        """Fetch all article URLs from archive pages in parallel using ThreadPoolExecutor."""
        # Create a ThreadPoolExecutor : scraping is very long if it is done page by page, so this enables to run parallel tasks and reduce the execution time by at least 3
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each archive URL
            futures = [executor.submit(self.get_article_urls, archive_url, archive_date) for archive_date, archive_url in self.archive_urls.items()]
            self.add_article_url(futures)


class Der_Spiegel_scraper(Archive_date)  :

    def __init__(self, params):
        super().__init__(params) # Call the parent class constructor with params

    def get_article_urls(self, archive_url, archive_date):
            """Get all article URLs and titles from a given archive page URL."""
            response = requests.get(archive_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_urls = []
            for article in soup.find_all('div', {'data-block-el': 'articleTeaser'}):
                link_tag = article.find('a', href=True)
                if link_tag:
                    url = link_tag['href']
                    title = link_tag.get('title', '')
                    article_urls.append({'link': url, 'title': title, 'date': archive_date})
            return article_urls

class El_Pais_scraper(Archive_date) :

    def __init__(self, params):
        super().__init__(params) # Call the parent class constructor with params

    def get_theme_from_url(self, url):
        # Define regex patterns to match both URL formats
        pattern1 = r"elpais\.com/diario/\d{4}/\d{2}/\d{2}/([^/]+)/"
        pattern2 = r"elpais\.com/([^/]+)/([^/]+)/\d{4}-\d{2}-\d{2}/[^/]+\.html"

        # Search for the first pattern in the URL
        match1 = re.search(pattern1, url)
        if match1:
            # Return the captured group (the theme)
            return match1.group(1)

        # Search for the second pattern in the URL
        match2 = re.search(pattern2, url)
        if match2:
            # Return the captured groups (themes)
            return match2.group(1)

        # If no pattern matches, return None
        return None


    def fetch_page_articles(self, url, archive_date):
        response = requests.get(url)
        article_urls = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup.find_all('h2', class_='c_t'):
                link_tag = tag.find('a', href=True)
                if link_tag:
                    theme = self.get_theme_from_url(link_tag['href'])
                    if theme not in self.params.get("themes_to_exclude", []):
                        title = link_tag.get_text(strip=True)
                        link = link_tag['href']
                        article_urls.append({'title': title, 'link': link, 'date': archive_date})
        return article_urls

    def get_article_urls(self, archive_url, archive_date):
        """Get all article URLs and titles from a given archive page URL."""
        page_num = 1
        article_urls = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            while True:
                url = f"{archive_url}{page_num}"
                futures.append(executor.submit(self.fetch_page_articles, url, archive_date))
                page_num += 1
                if len(futures) > 0 and futures[-1].result() == []:
                    break

            for future in concurrent.futures.as_completed(futures):
                article_urls.extend(future.result())

        return article_urls


class Le_Monde_scraper(Archive_date) :
    def __init__(self, params):
        super().__init__(params) # Call the parent class constructor with params


    def get_article_urls(self, archive_url, archive_date):
            response = requests.get(archive_url)
            soup = BeautifulSoup(response, "html.parser")
            article_urls = []
            # condition here : if no span icon__premium (abonnes)
            for tag in soup.find_all(class_="teaser"):
                link_tag = tag.find('a', href = True)
                # en-direct = video
                if link_tag:
                    link = link_tag['href']
                    title = link_tag.get('teaser_title', "")
                    if 'en-direct' not in link:
                      article_urls.append({'title': title, 'link': link, 'date' : archive_date})
            return article_urls




def scraper_factory(params):
    newspaper = params.get('newspaper')
    if newspaper == 'Der Spiegel':
        return Der_Spiegel_scraper(params)
    elif newspaper == 'El Pais':
        return El_Pais_scraper(params)
    elif newspaper == 'Le Monde':
        return Le_Monde_scraper(params)
    else:
        raise ValueError(f"Unknown newspaper: {newspaper}")

def main(params):
    scraper = scraper_factory(params)
    dataframe = scraper.process()
    return dataframe


#Remplacer par les thèmes à exclure en fonction de ceux qui existent sur le site
themes_to_exclude = themes_to_exclude = [
    "deportes",
    "ajedrez",
    "cultura",
    "icon",
    "ciencia",
    "elviajero",
    "comunicacion",
    "podcasts",
    "eps",
    "mexico", 
    "vintage",
    "espectaculo",
    "cine",
    "viajero", 
    "radiotv"]


#Remplir 
params = {
        'newspaper': 'El Pais',  # Le nom du journal que vous souhaitez scraper
        'url': 'https://elpais.com/hemeroteca/2008-06-11/',  # URL de modèle pour les archives
        'start_date': '19.05.2003',  # Date de début
        'end_date': '31.12.2023',  # Date de fin
        'themes_to_exclude' : themes_to_exclude
    }

if __name__ == "__main__":
    df = main(params)
       

