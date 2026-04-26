"""
Database connection and management module for MariaDB interactions.
Handles all database operations including connection management,
data insertion, retrieval, and table creation for financial data storage.
"""
import logging
import sys
import pandas as pd
from sqlalchemy import UniqueConstraint, create_engine, Column, Integer, String, Float, BigInteger, ForeignKey, DateTime, exc, select, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from src.tools.logger import logger

log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)

class MariadbConnector():
    """
    MariaDB database connector and manager class.
    
    Handles database connections, data operations, and table management
    for storing and retrieving financial time series data.
    
    Args:
        host (str): Database host address
        user (str): Database username
        password (str): Database password
        database (str): Database name
        port (int, optional): Database port number. Defaults to 3306
    """
    def __init__(self, host, user, password, database, port=3306):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.port = port
        self.engine = None
        self.connect_to_mariadb()
        self.retrieved_ticker_data = None
        #self.Session = None
        
    def test_connection(self):
        """
        Tests the database connection.
        
        Attempts to execute a simple query to verify that the database
        connection is active and functioning.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Try to execute a simple query
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except exc.OperationalError as e:
            log.log(f"Database server is not reachable: {e}", logging.ERROR)
            return False
        except Exception as e:
            log.log(f"Error testing database connection: {e}", logging.ERROR)
            return False

    def connect_to_mariadb(self):
        """
        Establishes connection to MariaDB database.
        
        Creates and configures database connection with appropriate timeout
        and reconnection settings. Initializes session maker and sets up models.
        
        Raises:
            Exception: If connection cannot be established
        """
        log.log("Establishing remote connection to MariaDB", logging.INFO)
        try:
            connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Enables automatic reconnection
                pool_recycle=3600,   # Recycle connections after 1 hour
                connect_args={
                    "connect_timeout": 5  # 5 seconds connection timeout
                }
            )
            # Test if we can actually connect
            if not self.test_connection():
                raise Exception("Could not establish database connection")
            self.Session = sessionmaker(bind=self.engine)
            self.setup_models()
            log.log("Established remote connection to MariaDB", logging.INFO)

        except exc.OperationalError as e:
            log.log(f"Error connecting to MariaDB - server may be down: {e}", logging.ERROR)
            sys.exit(1)
        except Exception as e:
            log.log(f"Error connecting to MariaDB: {e}", logging.ERROR)
            sys.exit(1)

    def get_session(self):
        """
        Creates a new database session.
        
        Tests connection and reconnects if necessary before
        returning a new session object.
        
        Returns:
            Session: New database session object
        """
        if not self.test_connection():
            self.connect_to_mariadb()
        return self.Session()

    def disconnect_from_mariadb(self):
        """
        Closes the database connection.
        
        Properly disposes of the engine and session objects
        to clean up database connections.
        """
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None
        log.log("Closed remote connection to MariaDB", logging.INFO)

    def setup_models(self):
        """
        Sets up database tables using SQLAlchemy models.
        
        Creates all necessary database tables if they don't exist,
        based on the defined model classes.
        
        Raises:
            Exception: If table creation fails
        """
        try:
            log.log("Creating database tables if they don't exist", logging.INFO)
            Base.metadata.create_all(self.engine)
            log.log("Database tables created successfully", logging.INFO)
        except Exception as e:
            log.log(f"Error creating database tables: {e}", logging.ERROR)
            raise

    def process_ticker_data(self, historical_data):
        """
        Preprocesses ticker data before database insertion.
        
        Handles missing values through interpolation and validates
        data quality before insertion.
        
        Args:
            historical_data (dict): Dictionary of ticker historical data
        """
        for key in historical_data:
            df = historical_data[key]
            #print(key)
            #print((df.head))
            nan_mean_print = None
            compliant = True
            for column in df.columns:
                nan_mean = df[column].isna().mean()
                if nan_mean>0.3:
                    compliant = False
                    nan_mean_print = nan_mean
                elif df[column].isnull().any():  # Only process columns with missing values
                    df.loc[:, column] = df[column].interpolate(method="polynomial", order=3)
                    # Forward fill for start gaps
                    df.loc[:, column] = df[column].ffill()
                    # Backward fill for end gaps
                    df.loc[:, column] = df[column].bfill()
            company_name = None
            sector = None
            #print(info_data[key])
            if compliant:
                self.insert_ticker_data(key, company_name, sector, df)
            else:
                log.log(f"Ticker '{key}' data has been deemed non-compliant: {nan_mean_print*100}% NaN", logging.WARNING)

    def insert_ticker_data(self, ticker_symbol, company_name, sector, df):
        """
        Inserts processed ticker data into database.
        
        Handles the creation of tickers, timestamps, and ticker data
        records with efficient bulk insertion.
        
        Args:
            ticker_symbol (str): Symbol identifying the ticker
            company_name (str): Name of the company
            sector (str): Company's sector
            df (pd.DataFrame): DataFrame containing ticker data
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        session = self.Session()
        try:
            # Get or create ticker in one operation
            ticker = session.query(Tickers).filter_by(symbol=ticker_symbol).first()
            if not ticker:
                ticker = Tickers(
                    symbol=ticker_symbol,
                    company_name=company_name,
                    sector=sector
                )
                session.add(ticker)
                session.flush()
            # Get all existing timestamps for the date range in one query
            date_range = df.index.tolist()
            existing_timestamps = {
                ts.data_time: ts
                for ts in session.query(Timestamps)
                            .filter(Timestamps.data_time.in_(date_range))
                            .all()
            }
            # Create missing timestamps
            new_timestamps = []
            for date in date_range:
                if date not in existing_timestamps:
                    timestamp = Timestamps(data_time=date)
                    session.add(timestamp)
                    new_timestamps.append(timestamp)
            if new_timestamps:
                session.flush()
                # Update existing_timestamps with new ones
                for ts in new_timestamps:
                    existing_timestamps[ts.data_time] = ts

            # Get existing ticker_data combinations to avoid duplicates
            existing_combinations = set(
                (td.ticker_id, td.timestamp_id)
                for td in session.query(TickerData)
                            .filter(TickerData.ticker_id == ticker.ticker_id)
                            .all()
            )

            # Prepare data for bulk insert
            data_to_insert = []
            for index, row in df.iterrows():
                timestamp = existing_timestamps[index]
                if (ticker.ticker_id, timestamp.timestamp_id) not in existing_combinations:
                    data_to_insert.append({
                        'ticker_id': ticker.ticker_id,
                        'timestamp_id': timestamp.timestamp_id,
                        'open_price': float(row['Open']),
                        'high_price': float(row['High']),
                        'low_price': float(row['Low']),
                        'close_price': float(row['Close']),
                        'volume': int(row['Volume'])
                    })

            # Bulk insert in batches
            batch_size = 1000
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i + batch_size]
                if batch:
                    session.execute(
                        TickerData.__table__.insert(),
                        batch
                    )
                    session.flush()

            session.commit()
            log.log(f"Successfully inserted all data for ticker {ticker_symbol}", logging.INFO)
            return True

        except SQLAlchemyError as e:
            session.rollback()
            log.log(f"Error inserting data for ticker {ticker_symbol}: {str(e)}", logging.ERROR)
            return False
        finally:
            session.close()

    def retrieve_ticker_data(self):
        """
        Retrieves all ticker data from database.
        
        Fetches and processes ticker data into a wide-format DataFrame
        with multi-level columns for different metrics.
        
        Returns:
            pd.DataFrame: Processed ticker data in wide format
        """
        # Create a session instance
        with self.Session() as session:
            query = select(
                Timestamps.data_time,
                Tickers.symbol,
                TickerData.open_price,
                TickerData.close_price,
                TickerData.high_price,
                TickerData.low_price,
                TickerData.volume
            ).join(
                TickerData, Timestamps.timestamp_id == TickerData.timestamp_id
            ).join(
                Tickers, TickerData.ticker_id == Tickers.ticker_id
            )
            # Execute query and load into DataFrame
            result = session.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=[
                'timestamp', 'symbol', 'open', 'close', 'high', 'low', 'volume'
            ])
            # Convert timestamp to datetime if it isn't already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Reshape the data using pivot
            # This creates MultiIndex columns with (symbol, metric)
            wide_df = df.pivot(
                index='timestamp',
                columns='symbol',
                values=['open', 'close', 'high', 'low', 'volume']
            )
            # Flatten the MultiIndex columns
            flat_columns = [
                f"{ticker}_{metric}"
                for metric, ticker in wide_df.columns
            ]
            wide_df.columns = flat_columns
            # Sort the columns alphabetically
            wide_df = wide_df.reindex(sorted(wide_df.columns), axis=1)
            # Sort index to ensure chronological order
            wide_df.sort_index(inplace=True)
            self.retrieved_ticker_data = wide_df
            return wide_df

    def return_ticker_data(self):
        """
        Returns the stored retrieved ticker data.
        
        Returns the cached version of previously retrieved ticker data
        without making a new database query.
        
        Returns:
            pd.DataFrame: Cached ticker data
        """
        return self.retrieved_ticker_data

Base = declarative_base()

class Tickers(Base):
    """
    SQLAlchemy model for the Tickers table.
    
    Stores basic information about each ticker including
    symbol, company name, and sector.
    """
    __tablename__ = 'Tickers'
    ticker_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False)
    company_name = Column(String(255))
    sector = Column(String(100))
    data = relationship('TickerData', back_populates='ticker')

class Timestamps(Base):
    """
    SQLAlchemy model for the Timestamps table.
    
    Stores unique timestamps for all ticker data points,
    preventing duplicate timestamp entries.
    """
    __tablename__ = 'Timestamps'
    timestamp_id = Column(Integer, primary_key=True, autoincrement=True)
    data_time = Column(DateTime, unique=True, nullable=False)
    data = relationship('TickerData', back_populates='timestamp')

class TickerData(Base):
    """
    SQLAlchemy model for the TickerData table.
    
    Stores actual ticker data including prices and volume,
    with foreign key relationships to Tickers and Timestamps.
    """
    __tablename__ = 'TickerData'
    data_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('Tickers.ticker_id'), nullable=False)
    timestamp_id = Column(Integer, ForeignKey('Timestamps.timestamp_id'), nullable=False)
    open_price = Column(Float)
    close_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    volume = Column(BigInteger)
    ticker = relationship('Tickers', back_populates='data')
    timestamp = relationship('Timestamps', back_populates='data')

    __table_args__ = (
        UniqueConstraint('ticker_id', 'timestamp_id', name='uq_ticker_timestamp'),
    )
