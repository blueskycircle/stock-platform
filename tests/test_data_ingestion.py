import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from library.data_ingestion import AlphaVantageIngestion


@pytest.fixture
def mock_engine():
    """Fixture to mock SQLAlchemy engine"""
    with patch("library.data_ingestion.create_engine") as mock:
        # Create a mock engine that we can use
        engine_mock = Mock()

        # Mock the connect method to return a context manager
        conn_mock = Mock()
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=conn_mock)
        context_manager.__exit__ = Mock(return_value=None)
        engine_mock.connect.return_value = context_manager

        # Configure the create_engine mock to return our engine_mock
        mock.return_value = engine_mock

        yield mock


@pytest.fixture
def mock_requests():
    """Fixture to mock HTTP requests"""
    with patch("library.data_ingestion.requests.get") as mock:
        yield mock


@pytest.fixture
def sample_stock_data():
    """Fixture providing sample stock data"""
    return {
        "Time Series (Daily)": {
            "2024-02-24": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. volume": "1000000",
            },
            "2024-02-23": {
                "1. open": "99.0",
                "2. high": "100.0",
                "3. low": "98.0",
                "4. close": "99.5",
                "5. volume": "900000",
            },
        }
    }


class TestAlphaVantageIngestion:

    # pylint: disable=redefined-outer-name
    def test_initialization(self, mock_engine):
        """Test class initialization and database setup"""
        ingestion = AlphaVantageIngestion()
        assert mock_engine.call_count == 2
        assert ingestion.base_url == "https://www.alphavantage.co/query"

    def test_fetch_stock_data_success(
        self, mock_engine, mock_requests, sample_stock_data
    ):
        """Test successful stock data fetching"""
        mock_response = Mock()
        mock_response.json.return_value = sample_stock_data
        mock_requests.return_value = mock_response

        ingestion = AlphaVantageIngestion()
        df = ingestion.fetch_stock_data("AAPL")

        # Verify timeout parameter was passed
        mock_requests.assert_called_once()
        _, kwargs = mock_requests.call_args
        assert kwargs.get("timeout") == 10

        # Existing assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
        assert df["symbol"].iloc[0] == "AAPL"
        assert isinstance(df["date"].iloc[0], datetime)

    def test_fetch_stock_data_api_error(self, mock_requests):
        """Test API error handling"""
        mock_requests.side_effect = Exception("API Error")

        ingestion = AlphaVantageIngestion()
        with pytest.raises(Exception, match="API Error"):
            ingestion.fetch_stock_data("AAPL")

    def test_store_data_success(self):
        """Test successful data storage"""
        data = {
            "date": [datetime(2024, 2, 24), datetime(2024, 2, 23)],
            "open": [100.0, 99.0],
            "high": [101.0, 100.0],
            "low": [99.0, 98.0],
            "close": [100.5, 99.5],
            "volume": [1000000, 900000],
            "symbol": ["AAPL", "AAPL"],
        }
        df = pd.DataFrame(data)

        with patch("pandas.DataFrame.to_sql") as mock_to_sql:
            ingestion = AlphaVantageIngestion()
            ingestion.store_data(df, "stock_prices")

            mock_to_sql.assert_called_once()
            _, kwargs = mock_to_sql.call_args
            assert kwargs["name"] == "stock_prices"
            assert kwargs["if_exists"] == "append"
            assert kwargs["index"] is False

    def test_update_stock_data(self, mock_requests, sample_stock_data):
        """Test the complete update flow"""
        # Set up API mock
        mock_response = Mock()
        mock_response.json.return_value = sample_stock_data
        mock_requests.return_value = mock_response

        # Set up database mocks
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = [pd.Timestamp("2024-02-23")]
        mock_connection.execute.return_value = mock_result

        # Create a context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)

        with patch("pandas.DataFrame.to_sql"), patch(
            "library.data_ingestion.create_engine"
        ) as mock_create_engine:

            # Set up the engine mock with proper context manager
            mock_engine = Mock()
            mock_engine.connect.return_value = context_manager
            mock_create_engine.return_value = mock_engine

            # Run the test
            ingestion = AlphaVantageIngestion()
            ingestion.update_stock_data(["AAPL", "GOOGL"])

            # Verify API calls
            assert mock_requests.call_count == 2
            calls = mock_requests.call_args_list
            assert calls[0][1]["params"]["symbol"] == "AAPL"
            assert calls[1][1]["params"]["symbol"] == "GOOGL"

            assert mock_connection.execute.call_count == 6

            # Get the SQL queries and parameters from the execute calls
            execute_calls = mock_connection.execute.call_args_list

            def get_sql_text(call):
                arg = call[0][0]
                if hasattr(arg, "text"):
                    sql_text = str(arg.text)
                else:
                    sql_text = str(arg)
                return " ".join(sql_text.split())  # Normalize whitespace

            def get_params(call):
                # Parameters are passed as keyword arguments in the second position
                if len(call[0]) > 1:
                    return call[0][1]
                # Or as a dictionary in the kwargs
                elif len(call[1]):
                    return call[1]
                return None

            queries = [get_sql_text(call) for call in execute_calls]
            params = [get_params(call) for call in execute_calls]

            # Verify query types and counts
            create_db_queries = [q for q in queries if "CREATE DATABASE" in q.upper()]
            create_table_queries = [q for q in queries if "CREATE TABLE" in q.upper()]
            select_queries = [
                q for q in queries if "SELECT MAX" in q.upper() and "date" in q.lower()
            ]

            # This helps identify any other queries that don't match the expected categories
            other_queries = [
                q
                for q in queries
                if "CREATE DATABASE" not in q.upper()
                and "CREATE TABLE" not in q.upper()
                and not ("SELECT MAX" in q.upper() and "date" in q.lower())
            ]

            assert (
                len(create_db_queries) == 1
            ), f"Expected 1 CREATE DATABASE query, got {len(create_db_queries)}"
            assert (
                len(create_table_queries) == 3
            ), f"Expected 3 CREATE TABLE queries, got {len(create_table_queries)}"
            assert (
                len(select_queries) == 2
            ), f"Expected 2 SELECT MAX(date) queries, got {len(select_queries)}"
            assert (
                len(other_queries) == 0
            ), f"Expected 0 other queries, got {len(other_queries)}"

            # Verify correct order of operations
            # Update these assertions if the order has changed
            assert (
                "CREATE DATABASE" in queries[0].upper()
            ), "First query should be CREATE DATABASE"

    @pytest.mark.parametrize(
        "invalid_data, expected_error",
        [
            ({}, "No time series data found for AAPL"),
            ({"Time Series (Daily)": {}}, "No time series data found for AAPL"),
            (
                {"Error Message": "Invalid API call"},
                "No time series data found for AAPL",
            ),
        ],
    )
    def test_fetch_stock_data_invalid_response(
        self, mock_requests, invalid_data, expected_error
    ):
        """Test handling of various invalid API responses"""
        mock_response = Mock()
        mock_response.json.return_value = invalid_data
        mock_requests.return_value = mock_response

        ingestion = AlphaVantageIngestion()
        with pytest.raises(KeyError, match=expected_error):
            ingestion.fetch_stock_data("AAPL")
