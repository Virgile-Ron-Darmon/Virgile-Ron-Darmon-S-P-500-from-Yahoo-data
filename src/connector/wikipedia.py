"""
A module for retrieving and processing S&P 500 company data from Wikipedia.
This module provides functionality to fetch current and historical S&P 500
company symbols from Wikipedia's API.
"""

import requests
from bs4 import BeautifulSoup
import logging
from src.tools.logger import logger

# Initialize logger
log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)


class WikipediaConnector:
    """
    A class to connect to Wikipedia and retrieve S&P 500 company information.
    
    This class fetches and processes data from Wikipedia's 'List of S&P 500 companies'
    page, extracting company symbols from both current and historical tables.
    """

    def __init__(self):
        """
        Initialize the WikipediaConnector with default settings and fetch initial data.
        
        The constructor sets up the Wikipedia page target and immediately retrieves
        and processes the relevant tables and symbols.
        """
        self.wikipedia_page = "List_of_S&P_500_companies"
        self.tables = self.retrieve_tables()
        self.symboles_current = []  # Stores current S&P 500 symbols
        self.symboles = self.symbole_importer()

    def return_symboles(self):
        """
        Return the list of processed S&P 500 symbols.
        
        Returns:
            list: A list of unique S&P 500 company symbols
        """
        return self.symboles

    def retrieve_tables(self):
        """
        Fetch and parse tables from the Wikipedia page using the MediaWiki API.
        
        Returns:
            list: BeautifulSoup objects representing the tables found on the page
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": self.wikipedia_page,
            "format": "json",
            "prop": "text",
            "contentmodel": "wikitext"
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extract the HTML content of the page
        html_content = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all tables with class 'wikitable'
        tables = soup.find_all("table", {"class": "wikitable"})
        return tables

    def symbole_importer(self):
        """
        Process the retrieved tables to extract and clean company symbols.
        
        This method handles both current and historical S&P 500 company symbols,
        removing duplicates and invalid entries.
        
        Returns:
            list: A cleaned list of unique company symbols
        """
        symbols = []
        
        # Process three different symbol columns from two tables
        for n in range(3):
            if n == 0:
                table_index = 0
                column_index = 0
            else:
                table_index = 1
                if n == 1:
                    column_index = 1
                else:
                    column_index = 3
                    
            table = self.tables[table_index]
            
            # Extract symbols from the specified column
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                symbols.append(cells[column_index].get_text(strip=True))
                if table_index == 1:
                    self.symboles_current.append(cells[column_index].get_text(strip=True))

        # Clean and process current symbols
        self.symboles_current = list(set(symbols))
        self.symboles_current = [i for i in symbols if len(i) > 1]
        self.symboles_current = [' '.join(filter(str.isupper, a.split())) for a in symbols]
        self.symboles_current.remove('')

        # Clean and process all symbols
        symbols = list(set(symbols))
        symbols = [i for i in symbols if len(i) > 1]
        symbols = [' '.join(filter(str.isupper, a.split())) for a in symbols]
        symbols.remove('')
        
        return symbols