"""
Data post-processing module for financial time series data.
Provides functionality for cleaning and filtering DataFrame-based
financial data, particularly handling NaN values.
"""
import logging
from src.tools.logger import logger

log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)


class DfPost:
    """
    Data post-processing class for financial DataFrames.
    
    Implements methods for filtering and cleaning financial time series data,
    with particular focus on handling missing values and maintaining data quality.
    """
    def __init__(self):
        pass
        #self.processedDf = self.filter_dataframe_by_nan(df, 0.9, 0.9)
        #a, b, c, d =self.preprocess_data(self.processedDf)

    @staticmethod
    def filter_dataframe_by_nan(df, column_threshold, row_threshold):
        """
        Filters DataFrame based on NaN value thresholds.
        
        Removes rows and columns that exceed specified thresholds for
        missing values to maintain data quality.
        
        Args:
            df (pd.DataFrame): Input DataFrame to filter
            column_threshold (float): Maximum allowed NaN ratio per column
            row_threshold (float): Maximum allowed NaN ratio per row
            
        Returns:
            pd.DataFrame: Filtered DataFrame meeting threshold requirements
        """

        # Remove rows where the percentage of NaN is above the upper threshold
        row_non_nan_percentage = 1 - df.isna().mean(axis=1)

        df = df[row_non_nan_percentage >= row_threshold]

        # Remove columns where the percentage of NaN is above the upper threshold
        column_non_nan_percentage = 1 - df.isna().mean()

        df = df.loc[:, column_non_nan_percentage >= column_threshold]
        return df
