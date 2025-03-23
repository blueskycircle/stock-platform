import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class StockAnalysis:
    """
    Class for analyzing and visualizing stock data from the database
    """

    def __init__(self, engine=None):
        """
        Initialize the StockAnalysis class

        Parameters:
        -----------
        engine : SQLAlchemy engine, optional
            SQLAlchemy engine to connect to the database. If not provided,
            a new engine will be created using environment variables.
        """
        if engine is None:
            db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "database": os.getenv("DB_NAME"),
            }
            self.engine = create_engine(
                f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}/{db_config['database']}"
            )
        else:
            self.engine = engine

        # Set default plotting style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 12

    def fetch_stock_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch stock data from the database

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. If None, fetches data up to current date.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing stock data
        """
        query = "SELECT * FROM stock_prices WHERE symbol = :symbol"
        params = {"symbol": symbol}

        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY date ASC"

        try:
            df = pd.read_sql(text(query), self.engine, params=params)

            if df.empty:
                logger.warning("No data found for %s", symbol)
                return None

            # Convert date to datetime if it's not already
            df["date"] = pd.to_datetime(df["date"])

            return df

        except (SQLAlchemyError, pd.errors.DatabaseError) as e:
            logger.error("Error fetching data for %s: %s", symbol, str(e))
            return None
        except ValueError as e:
            # Handle date conversion errors
            logger.error("Error processing dates for %s: %s", symbol, str(e))
            return None
        except KeyError as e:
            # Handle missing column errors
            logger.error("Missing column in data for %s: %s", symbol, str(e))
            return None

    def plot_price_history(
        self, symbol, start_date=None, end_date=None, show_volume=True, save_path=None
    ):
        """
        Plot the price history of a stock

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        show_volume : bool, default True
            Whether to include volume in the plot
        save_path : str, optional
            If provided, saves the plot to the specified path

        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        df = self.fetch_stock_data(symbol, start_date, end_date)

        if df is None or len(df) == 0:
            logger.error("No data available for %s", symbol)
            return None

        # Create figure with one or two subplots depending on volume
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        # Plot price
        ax1.plot(df["date"], df["close"], "b-", linewidth=2)

        # Add moving averages
        if len(df) >= 50:
            df["MA50"] = df["close"].rolling(window=50).mean()
            ax1.plot(
                df["date"],
                df["MA50"],
                "r-",
                linewidth=1.5,
                alpha=0.8,
                label="50-Day MA",
            )

        if len(df) >= 200:
            df["MA200"] = df["close"].rolling(window=200).mean()
            ax1.plot(
                df["date"],
                df["MA200"],
                "g-",
                linewidth=1.5,
                alpha=0.8,
                label="200-Day MA",
            )

        # Add price range shading
        ax1.fill_between(df["date"], df["low"], df["high"], color="skyblue", alpha=0.3)

        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.2f}"))

        # Set title and labels
        ax1.set_title(f"{symbol} Stock Price History")
        ax1.set_ylabel("Price ($)")

        # Add legend
        if len(df) >= 50:
            ax1.legend()

        # Add grid
        ax1.grid(True, alpha=0.3)

        # Format x-axis dates
        plt.xticks(rotation=45)

        # Add volume subplot if requested
        if show_volume:
            # Plot volume bars
            ax2.bar(df["date"], df["volume"], color="gray", alpha=0.5)
            ax2.set_ylabel("Volume")
            ax2.grid(True, alpha=0.3)

            # Format y-axis with M/B suffixes for millions/billions
            ax2.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda y, _: f"{y/1e6:.0f}M" if y < 1e9 else f"{y/1e9:.1f}B"
                )
            )

        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved price history plot to %s", save_path)

        return fig

    def plot_returns_distribution(
        self, symbol, start_date=None, end_date=None, period="daily", save_path=None
    ):
        """
        Plot the distribution of stock returns

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        period : str, default 'daily'
            Period for calculating returns ('daily', 'weekly', 'monthly')
        save_path : str, optional
            If provided, saves the plot to the specified path

        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        df = self.fetch_stock_data(symbol, start_date, end_date)

        if df is None or len(df) == 0:
            logger.error("No data available for %s", symbol)
            return None

        # Set date as index
        df.set_index("date", inplace=True)

        # Calculate returns based on the specified period
        if period == "daily":
            returns = df["close"].pct_change().dropna()
            period_name = "Daily"
        elif period == "weekly":
            returns = df["close"].resample("W").last().pct_change().dropna()
            period_name = "Weekly"
        elif period == "monthly":
            returns = df["close"].resample("ME").last().pct_change().dropna()
            period_name = "Monthly"
        else:
            logger.error(
                "Invalid period: %s. Use 'daily', 'weekly', or 'monthly'", period
            )
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram
        sns.histplot(returns, bins=50, kde=True, ax=ax)

        # Add a vertical line at 0
        plt.axvline(x=0, color="r", linestyle="--", alpha=0.7)

        # Add normal distribution fit if scipy is available
        try:
            import scipy.stats as stats

            if len(returns) >= 30:
                mu, std = stats.norm.fit(returns)
                x = np.linspace(returns.min(), returns.max(), 100)
                p = stats.norm.pdf(x, mu, std)
                plt.plot(x, p, "k", linewidth=2)
                plt.title(
                    f"{period_name} Returns Distribution for {symbol} (μ={mu:.4f}, σ={std:.4f})"
                )
            else:
                plt.title(f"{period_name} Returns Distribution for {symbol}")
        except ImportError:
            logger.warning(
                "scipy.stats not available, skipping normal distribution fit"
            )
            plt.title(f"{period_name} Returns Distribution for {symbol}")

        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))

        # Add labels
        plt.xlabel(f"{period_name} Return")
        plt.ylabel("Frequency")

        # Add statistics
        stats_text = (
            f"Mean: {returns.mean():.2%}\n"
            f"Median: {returns.median():.2%}\n"
            f"Std Dev: {returns.std():.2%}\n"
            f"Min: {returns.min():.2%}\n"
            f"Max: {returns.max():.2%}"
        )

        plt.annotate(
            stats_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            verticalalignment="top",
        )

        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved returns distribution plot to %s", save_path)

        return fig

    def create_performance_dashboard(
        self, symbol, start_date=None, end_date=None, save_path=None
    ):
        """
        Create a comprehensive performance dashboard for a stock

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        save_path : str, optional
            If provided, saves the plot to the specified path

        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """

        df = self.fetch_stock_data(symbol, start_date, end_date)

        if df is None or len(df) == 0:
            logger.error("No data available for %s", symbol)
            return None

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{symbol} Performance Dashboard", fontsize=20)

        # Plot 1: Price history
        ax1 = axes[0, 0]
        ax1.plot(df["date"], df["close"], "b-", linewidth=2)

        # Add moving averages
        if len(df) >= 50:
            df["MA50"] = df["close"].rolling(window=50).mean()
            ax1.plot(
                df["date"],
                df["MA50"],
                "r-",
                linewidth=1.5,
                alpha=0.8,
                label="50-Day MA",
            )

        if len(df) >= 200:
            df["MA200"] = df["close"].rolling(window=200).mean()
            ax1.plot(
                df["date"],
                df["MA200"],
                "g-",
                linewidth=1.5,
                alpha=0.8,
                label="200-Day MA",
            )

        ax1.set_title("Price History")
        ax1.set_ylabel("Price ($)")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.2f}"))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume
        ax2 = axes[0, 1]
        ax2.bar(df["date"], df["volume"], color="gray", alpha=0.5)
        ax2.set_title("Trading Volume")
        ax2.set_ylabel("Volume")
        ax2.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y/1e6:.0f}M" if y < 1e9 else f"{y/1e9:.1f}B")
        )
        ax2.grid(True, alpha=0.3)

        # Plot 3: Daily returns
        ax3 = axes[1, 0]
        df["daily_return"] = df["close"].pct_change()
        ax3.bar(df["date"], df["daily_return"], color="green", alpha=0.5)
        ax3.axhline(y=0, color="r", linestyle="--", alpha=0.7)
        ax3.set_title("Daily Returns")
        ax3.set_ylabel("Return")
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax3.grid(True, alpha=0.3)

        # Plot 4: Return distribution
        ax4 = axes[1, 1]
        sns.histplot(df["daily_return"].dropna(), bins=50, kde=True, ax=ax4)
        ax4.axvline(x=0, color="r", linestyle="--", alpha=0.7)
        ax4.set_title("Return Distribution")
        ax4.set_xlabel("Daily Return")
        ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax4.grid(True, alpha=0.3)

        # Add summary statistics - now with better handling of data
        # Make a copy of the dataframe for statistics to avoid modifying the original
        stats_df = df.copy()

        # Calculate returns only on the data we have
        valid_prices = stats_df[stats_df["close"].notnull()]

        # Use first and last valid prices
        if not valid_prices.empty:
            first_valid_idx = valid_prices.index[0]
            last_valid_idx = valid_prices.index[-1]

            start_date = valid_prices.loc[first_valid_idx, "date"]
            end_date = valid_prices.loc[last_valid_idx, "date"]

            start_price = valid_prices.loc[first_valid_idx, "close"]
            end_price = valid_prices.loc[last_valid_idx, "close"]

            # Calculate price return (not total return since we don't have dividend data)
            price_return = (end_price / start_price) - 1

            # Calculate days and years
            days = (end_date - start_date).days
            years = max(
                days / 365.25, 0.01
            )  # Use 365.25 for leap years, minimum of ~3.65 days

            # Calculate annualized return
            annualized_return = (1 + price_return) ** (1 / years) - 1

            # Calculate volatility using clean data
            daily_returns = valid_prices["close"].pct_change().dropna()
            volatility = daily_returns.std() * (252**0.5)  # Annualized volatility

            # Calculate Sharpe ratio (assuming 0 risk-free rate for simplicity)
            sharpe = annualized_return / volatility if volatility > 0 else 0

            # Format the display text - now with note about price return only
            stats_text = (
                f"Price Return: {price_return:.2%}\n"  # Changed from Total Return to Price Return
                f"Annualized Return: {annualized_return:.2%}\n"
                f"Annualized Volatility: {volatility:.2%}\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
                f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)"
            )
        else:
            stats_text = "Insufficient data for statistics"

        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # Add a note about price return vs total return
        fig.text(
            0.5,
            0.01,
            "Note: Returns shown are price returns only (excluding dividends)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.97])

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved performance dashboard to %s", save_path)

        return fig

    def compare_stocks(
        self, symbols, start_date=None, end_date=None, normalized=True, save_path=None
    ):
        """
        Compare performance of multiple stocks

        Parameters:
        -----------
        symbols : list
            List of stock ticker symbols
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        normalized : bool, default True
            If True, normalize all prices to start at 100
        save_path : str, optional
            If provided, saves the plot to the specified path

        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if not symbols or len(symbols) < 1:
            logger.error("At least one symbol is required for comparison")
            return None

        # Create an empty DataFrame for comparison
        comparison_df = pd.DataFrame()

        # Fetch data for each symbol
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                # Add the close price to the comparison DataFrame
                comparison_df[symbol] = df.set_index("date")["close"]
            else:
                logger.warning("No data available for %s", symbol)

        if comparison_df.empty:
            logger.error("No data available for any of the specified symbols")
            return None

        # Normalize if requested
        if normalized:
            comparison_df = comparison_df / comparison_df.iloc[0] * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each stock
        for symbol in comparison_df.columns:
            ax.plot(
                comparison_df.index, comparison_df[symbol], linewidth=2, label=symbol
            )

        # Set title and labels
        if normalized:
            ax.set_title("Normalized Stock Price Comparison (Base=100)")
            ax.set_ylabel("Normalized Price")
        else:
            ax.set_title("Stock Price Comparison")
            ax.set_ylabel("Price ($)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.2f}"))

        # Add legend and grid
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Calculate performance statistics
        returns = comparison_df.iloc[-1] / comparison_df.iloc[0] - 1
        sorted_returns = returns.sort_values(ascending=False)

        # Add text box with performance summary
        stats_text = "Total Return:\n" + "\n".join(
            [f"{symbol}: {ret:.2%}" for symbol, ret in sorted_returns.items()]
        )

        plt.annotate(
            stats_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            verticalalignment="top",
        )

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved stock comparison plot to %s", save_path)

        return fig

    def get_distinct_values(self, column_name, table_name="stock_prices"):
        """
        Get all unique values from a specific column in the database

        Parameters:
        -----------
        column_name : str
            Name of the column to get distinct values from
        table_name : str, default "stock_prices"
            Name of the table to query

        Returns:
        --------
        list
            List of unique values in the specified column
        """

        query = text(
            f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name} ASC"
        )

        try:
            with self.engine.connect() as connection:
                result = connection.execute(query)
                distinct_values = [row[0] for row in result]

            return distinct_values

        except SQLAlchemyError as e:
            logger.error(
                "Database error getting distinct values for %s: %s", column_name, str(e)
            )
            return []
        except (ValueError, TypeError) as e:
            logger.error(
                "Data error getting distinct values for %s: %s", column_name, str(e)
            )
            return []
