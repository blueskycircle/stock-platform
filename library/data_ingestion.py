import requests
import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from dotenv import load_dotenv
import sqlalchemy.exc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlphaVantageIngestion:
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }

        # Create database if it doesn't exist
        temp_engine = create_engine(
            f"mysql+mysqlconnector://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}"
        )
        with temp_engine.connect() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            )

        # Create engine with database selected
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}/{self.db_config['database']}"
        )

    def fetch_stock_data(self, symbol, function="TIME_SERIES_DAILY"):
        """
        Fetch stock data from Alpha Vantage API
        """
        try:
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            # Add timeout parameter to prevent hanging
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data or not data[time_series_key]:
                raise KeyError(f"No time series data found for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            df.index = pd.to_datetime(df.index)

            # Rename columns
            df.columns = [col.split(". ")[1] for col in df.columns]
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)

            # Add symbol column
            df["symbol"] = symbol

            return df

        except Exception as e:
            # Use lazy % formatting instead of f-string
            logger.error("Error fetching data for %s: %s", symbol, str(e))
            raise

    def store_data(self, df, table_name):
        """
        Store data in MySQL database with update/insert logic
        """
        try:
            # Create table with proper schema if it doesn't exist
            create_table_query = text(
                f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE NOT NULL,
                open DECIMAL(10,2) NOT NULL,
                high DECIMAL(10,2) NOT NULL,
                low DECIMAL(10,2) NOT NULL,
                close DECIMAL(10,2) NOT NULL,
                volume BIGINT NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                PRIMARY KEY (date, symbol)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            )

            with self.engine.connect() as connection:
                connection.execute(create_table_query)
                connection.commit()

            # Convert numeric columns to appropriate types
            numeric_columns = ["open", "high", "low", "close"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"], downcast="integer")

            # Store the data
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists="append",
                index=False,
                method="multi",
            )

            logger.info(
                "Successfully stored %d records for %s",
                len(df),
                df['symbol'].iloc[0]
            )

        except Exception as e:
            logger.error("Error storing data: %s", str(e))
            raise

    def update_stock_data(self, symbols, table_name="stock_prices"):
        """Update stock data for multiple symbols"""
        for symbol in symbols:
            try:
                logger.info("Fetching data for %s", symbol)
                df = self.fetch_stock_data(symbol)

                try:
                    # Check for existing data
                    query = text(
                        f"SELECT MAX(date) as last_date FROM {table_name} WHERE symbol = :symbol"
                    )
                    with self.engine.connect() as connection:
                        result = connection.execute(query, {"symbol": symbol}).fetchone()
                    last_date = pd.Timestamp(result[0]) if result[0] else None

                    if last_date is not None:
                        # Filter only new data using pandas datetime
                        df['date'] = pd.to_datetime(df['date'])
                        df = df[df['date'] > last_date]

                except (sqlalchemy.exc.ProgrammingError, sqlalchemy.exc.OperationalError) as db_error:
                    logger.warning("Database error for %s: %s", symbol, str(db_error))

                if not df.empty:
                    self.store_data(df, table_name)
                    logger.info("Added %d new records for %s", len(df), symbol)
                else:
                    logger.info("No new data to add for %s", symbol)

            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                logger.error("Error processing %s: %s", symbol, str(e))
                continue
