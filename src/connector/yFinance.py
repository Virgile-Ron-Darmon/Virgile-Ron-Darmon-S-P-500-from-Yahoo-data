"""
A module for fetching and processing financial data from Yahoo Finance.
This module provides functionality to batch download historical price data
for multiple stock tickers and store it in a database.
"""

import yfinance as yf
import logging
import time
from src.tools.logger import logger

# Initialize logger
log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)


class YfinanceConnector:
    """
    A class to manage connections to Yahoo Finance and handle data retrieval.
    
    This class provides batch processing capabilities for downloading historical
    stock data, with built-in rate limiting and database storage functionality.
    """
    
    def __init__(self, mariadb, start='2017-04-01', end='2024-04-01', interval='1d'):
        """
        Initialize the YfinanceConnector with specified parameters.
        
        Args:
            mariadb: A database connection object for storing retrieved data
            start (str): Start date for historical data (default: '2017-04-01')
            end (str): End date for historical data (default: '2024-04-01')
            interval (str): Data interval (default: '1d' for daily)
        """
        self.mariadb = mariadb
        self.start = start
        self.end = end
        self.interval = interval
        self.ticker_list = None
        self.historical_data = None
        self.info_data = None

    def importer_t(self, ticker_list):
        """
        Import data for a list of tickers.
        
        Args:
            ticker_list (list): List of stock ticker symbols to process
        """
        self.ticker_list = ticker_list
        self.historical_data, self.info_data = self.importer()

    def return_ticker_history(self):
        """
        Return the collected historical data.
        
        Returns:
            dict: Historical price data for all processed tickers
        """
        return self.historical_data

    def importer(self):
        """
        Main import function that handles batch processing of ticker data.
        
        This method implements rate limiting and batch processing to avoid
        overwhelming the Yahoo Finance API. It also handles database storage
        of retrieved data.
        
        Returns:
            tuple: (historical_data, info_data) containing the retrieved data
        """
        log.log("Collection data for " + str(len(self.ticker_list)) + " Tickers", logging.INFO)
        
        # Configuration for batch processing
        max_batch = 5      # Maximum number of tickers per batch
        delay = 15         # Delay between batches in seconds
        import_start = time.time()

        info_data = {}
        
        # Process tickers in batches
        for i in range(0, len(self.ticker_list), max_batch):
            historical_data = {}

            # Prepare current batch
            batch = self.ticker_list[i:i + max_batch]
            ticker_symbole = " ".join(batch)
            log.log("Collection data from following Tickers:\n    -> " + ticker_symbole, logging.INFO)
            
            # Fetch data for current batch
            tickers = yf.Tickers(ticker_symbole)
            data = tickers.history(start=self.start, end=self.end, interval=self.interval)

            # Process each ticker in the batch
            for ticker in batch:
                ticker_data = data.loc[:, data.columns.get_level_values(1) == ticker]

                if not ticker_data.empty:
                    ticker_data.columns = ticker_data.columns.droplevel(1)
                    # Only store data if there are valid price entries
                    if not ticker_data['Close'].isna().all():
                        historical_data[ticker] = ticker_data

            # Store data in database
            log.log("Storing data in Database", logging.INFO)
            operation_start = time.time()
            self.mariadb.process_ticker_data(historical_data)
            operation_end = time.time()

            # Handle rate limiting
            operation_time = operation_end - operation_start
            operation_await = max(0, delay - operation_time)
            log.log(f"Operation lasted {operation_time:.2f}/{delay:.2f} seconds, {operation_await:.2f} seconds delay required", logging.INFO)
            time.sleep(operation_await)

        # Log total import time
        import_end = time.time()
        import_time = import_end - import_start
        log.log(f"Import lasted {import_time/60:.2f} minutes", logging.INFO)

        return historical_data, info_data