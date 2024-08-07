#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:34:59 2024

@author: cocoticota
"""

from bs4 import BeautifulSoup
import requests
import logging
import re
import pandas as pd

# Configurer les logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GetArticleText:
    
    def __init__(self, list_of_links, paragraph_class="article__paragraph"):
        self.links = list_of_links
        self.paragraph_class = paragraph_class
        self.text = self.extract_text()
        
    def extract_text(self):
        results = []
        for link in self.links:
            try:
                logging.info(f"Ouverture de la page : {link}")
                soup = self.open_page(link)
                article_text = self.read_text(soup)
                results.append(article_text)
            except Exception as e:
                logging.error(f"Erreur lors du traitement du lien {link}: {e}")
        return results
            
    def open_page(self, link):
        try:
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except requests.RequestException as e:
            logging.error(f"Erreur lors de l'ouverture de la page {link}: {e}")
            raise
        
    def read_text(self, soup):
        article_text = []
        for tag in soup.find_all("p", class_=self.paragraph_class):
            article_text.append(tag.get_text(strip=True))
        article_text = " ".join(article_text)  # Joindre les paragraphes en un seul texte
        cleaned_text = re.sub(r"\\'", "'", article_text)  # Nettoyer résidus HTML
        return cleaned_text

def main(data, condition_column, link_column): 
    # Filtrer les lignes où 'condition_column' est égal à 1
    filtered_data = data[data[condition_column] == 1]
    list_of_links = filtered_data[link_column].to_list()
    
    # Extraire les textes des articles
    extractor = GetArticleText(list_of_links)
    results = extractor.text
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results, columns=['Full_Text'], index=filtered_data.index)
    
    # Joindre les résultats au DataFrame original
    data = data.join(results_df)
    
    return data

# Exemple d'utilisation
# Assurez-vous d'avoir un DataFrame 'df' avec les colonnes appropriées avant d'appeler 'main'
# df = pd.DataFrame({'condition_column': [1, 0, 1], 'link_column': ['link1', 'link2', 'link3']})
# result_df = main(df, 'condition_column', 'link_column')
# print(result_df)
