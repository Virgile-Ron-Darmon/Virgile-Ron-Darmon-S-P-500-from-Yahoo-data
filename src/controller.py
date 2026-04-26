"""
Main controller module for the financial data analysis system.
Orchestrates data loading, processing, model training, and visualization
components of the application.
"""
import logging
import random
import time
import numpy as np
import yaml
from src.view.plotter import Plotter
from src.tools.logger import logger
from src.connector.mariadb import MariadbConnector
from src.connector.yFinance import YfinanceConnector
from src.connector.wikipedia import WikipediaConnector
from src.model.df_post import DfPost
from src.model.model_predict_pytorch import TimeSeriesPredictor

log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)


class Controller():
    """
    Main controller class coordinating all system operations.
    
    Handles configuration loading, database connections, data processing,
    model training, and visualization generation for the entire system.
    """
    def __init__(self):
        # Path to the configuration file
        config_file = './config.yaml'
        self.load_config(config_file)
        self.mariadb_c = None

    def load_config(self, file_path):
        """
        Loads and validates system configuration from a YAML file.
        
        Args:
            file_path (str): Path to the configuration YAML file
            
        Creates default configuration if file is missing or invalid.
        """
        default_config = {}
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file) or {}

                if not isinstance(config, dict):
                    raise ValueError("Config file does not contain a valid YAML dictionary")
                log.log("Config loaded successfully", logging.INFO)

        except FileNotFoundError:
            log.log("Config file not found", logging.WARNING)
            log.log("Creating config.yaml with default values", logging.INFO)
            config = default_config

            with open(file_path, "w") as file:
                yaml.safe_dump(default_config, file, default_flow_style=False)
            log.log("Default config created and loaded", logging.INFO)

        except (yaml.YAMLError, ValueError) as e:
            log.log(f"Error loading config: {e}", logging.ERROR)
            log.log("Loading default configuration", logging.INFO)
            config = default_config

        # database variables
        self.config_host = config.get('Host', '86.15.2.245')
        self.config_port = int(config.get('Port', '3306'))
        self.config_user = config.get('User', '0')
        self.config_password = config.get('Password', '0')
        self.config_db = config.get('DB', '0')

        self.config_start = config.get('start', "2017-04-01")
        self.config_end = config.get('end', "2024-04-01")
        self.config_cutoff = config.get('cutoff', '2024-02-01')
        self.config_target_column = config.get('target', '^GSPC_close')
        self.config_demo = bool(config.get('demo', True))

        self.plotter = Plotter(self.config_demo)

        self.config_major_indexes = config.get('major_indexes', [])
        self.config_international_indexes = config.get('international_indexes', [])
        self.config_cocommodity_indexes = config.get('commodity_indexes', [])
        self.config_crypto_indexes = config.get('crypto_indexes', [])
        self.config_other_indexes = config.get('other_indexes', [])
        self.indexes_sumbols = (self.config_major_indexes+
                              self.config_international_indexes+
                              self.config_cocommodity_indexes+
                              self.config_crypto_indexes+
                              self.config_other_indexes)

        self.model = TimeSeriesPredictor(
            sequence_length=21,
            n_components=0.95,
            hidden_size=64,
            num_layers=2,
            batch_size=32,
            num_epochs=100
        )
        self.model_config = {
            'sequence_length': config.get('model_sequence_length', 21),
            'n_components': config.get('model_n_components', 0.95),
            'hidden_size': config.get('model_hidden_size', 64),
            'num_layers': config.get('model_num_layers', 2),
            'batch_size': config.get('model_batch_size', 32),
            'num_epochs': config.get('model_num_epochs', 100),
            'target_column': config.get('model_target_column', '^GSPC_close'),
            'cutoff_date': config.get('model_cutoff_date', '2024-02-01')
        }

    def run(self):
        """
        Executes the main system workflow.
        
        Orchestrates the entire process including:
        - Database connection
        - Data import
        - Data processing
        - Model training
        - Prediction generation
        - Visualization creation
        """
        # Previous data loading code remains the same until the model training part
        log.log("===== Starting: Data Storage Stage =====", logging.INFO)
        self.mariadb_c = MariadbConnector(
            self.config_host,
            self.config_user,
            self.config_password,
            self.config_db,
            self.config_port
        )

        log.log("===== Starting: Data Import Stage =====", logging.INFO)
        wikipedia_c = WikipediaConnector()
        ticker_symbole = wikipedia_c.return_symboles()
        all_symboles = self.indexes_sumbols + ticker_symbole
        yfinance_c = YfinanceConnector(self.mariadb_c, self.config_start, self.config_end, '1d')
        demo = []

        m = 15
        if self.config_demo:
            log.log(f"---> You are running in DEMO mode, {m} Tickers' data will be downloaded to meet execution time requirements", logging.DEBUG)
            for n in range(m):
                demo.append(random.choice(all_symboles))
            yfinance_c.importer_t(demo)
        else:
            yfinance_c.importer_t(all_symboles)

        log.log("===== Starting: Data Import Stage =====", logging.INFO)
        start_time = time.time()
        self.mariadb_c.retrieve_ticker_data()
        end_time = time.time()
        execution_time = end_time - start_time
        log.log(f"Import lasted {execution_time/60:.2f} minutes", logging.INFO)

        log.log("===== Starting: Data Processing and Model Stage =====", logging.INFO)
        df_p = DfPost()

        # Define model parameters
        sequence_length = self.model_config['sequence_length']

        # Process the data
        processed_df = df_p.filter_dataframe_by_nan(self.mariadb_c.return_ticker_data().copy(), 1, 0.9)
        self.plotter.dataframe_to_nan_image(self.mariadb_c.return_ticker_data(), processed_df)

        # Train the model
        start_time = time.time()
        self.model.train(processed_df.copy(), self.config_target_column, self.config_cutoff)
        end_time = time.time()
        execution_time = end_time - start_time
        log.log(f"Model Training lasted {execution_time/60:.2f} minutes", logging.INFO)

        self.plotter.plot_data_processing_pipeline(
            self.mariadb_c.return_ticker_data(),
            processed_df.copy(),
            self.model.get_processing_data(),
            self.config_target_column
        )

        # Get test data and parameters
        test_data = processed_df[processed_df.index >= self.config_cutoff].copy()
        test_dates = test_data.index
        real_values = test_data[self.config_target_column].values
        num_prediction_days = len(test_data)

        # Get the last sequence_length days before cutoff for initial prediction
        last_real_sequence = processed_df[processed_df.index < self.config_cutoff].iloc[-sequence_length:]

        # Approach 1: Predict all days at once
        all_at_once_predictions = self.model.predict_multiple_days(
            last_real_sequence,
            num_prediction_days,
            self.config_target_column
        )
        # Approach 2: Predict one day at a time using actual data
        one_by_one_predictions = []
        for i in range(num_prediction_days):
            current_date = test_dates[i]
            start_idx = processed_df.index.get_loc(current_date) - sequence_length + 1
            end_idx = processed_df.index.get_loc(current_date) + 1
            current_sequence = processed_df.iloc[start_idx:end_idx].copy()

            # Make single day prediction
            prediction = self.model.predict(current_sequence.drop(columns=[self.config_target_column]))
            one_by_one_predictions.append(prediction)

        # Calculate error metrics
        rmse_all_at_once = np.sqrt(np.mean((real_values - all_at_once_predictions) ** 2))
        rmse_one_by_one = np.sqrt(np.mean((real_values - one_by_one_predictions) ** 2))

        log.log(f"RMSE for all-at-once predictions: {rmse_all_at_once:.4f}", logging.INFO)
        log.log(f"RMSE for one-by-one predictions: {rmse_one_by_one:.4f}", logging.INFO)

        if len(test_dates) == 0 and len(real_values) == 0:
            log.log("No test data available for plotting AT ALL", logging.WARNING)
        elif len(test_dates) == 0:
            log.log("No test data available for plotting DATES", logging.WARNING)
        elif len(real_values) == 0:
            log.log("No test data available for plotting REAL VAL", logging.WARNING)
        else:
            self.plotter.plot_predictions(
                test_dates,
                real_values,
                one_by_one_predictions,
                all_at_once_predictions,
                self.config_target_column
            )

        self.plotter.giga_plotter_function()

        self.mariadb_c.disconnect_from_mariadb()
