import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot
import click
from library.data_ingestion import AlphaVantageIngestion
from library.data_analysis import StockAnalysis
import logging
from sqlalchemy import text
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Command line interface for stock data ingestion and analysis"""


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--table-name",
    "-t",
    default="stock_prices",
    help="Name of the database table to store the data",
)
def fetch(symbols, table_name):
    """
    Fetch and store stock data for one or more symbols.

    Usage: python cli.py fetch AAPL MSFT GOOGL
    """
    try:
        ingestion = AlphaVantageIngestion()
        ingestion.update_stock_data(list(symbols), table_name)
        click.echo(f"Successfully processed data for symbols: {', '.join(symbols)}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("symbol")
@click.option(
    "--table-name",
    "-t",
    default="stock_prices",
    help="Name of the database table to query",
)
@click.option("--limit", "-l", default=5, help="Number of records to show")
def show(symbol, table_name, limit):
    """
    Show the latest stock data for a symbol.

    Usage: python cli.py show AAPL --limit 10
    """
    try:
        ingestion = AlphaVantageIngestion()
        query = text(
            f"""
        SELECT date, open, high, low, close, volume 
        FROM {table_name} 
        WHERE symbol = :symbol
        ORDER BY date DESC
        LIMIT :limit
        """
        )

        with ingestion.engine.connect() as connection:
            result = connection.execute(
                query, {"symbol": symbol, "limit": limit}
            ).fetchall()

        if not result:
            click.echo(f"No data found for symbol: {symbol}")
            return

        # Print results in a formatted table
        click.echo(f"\nLatest {limit} records for {symbol}:")
        click.echo("-" * 80)
        click.echo(
            f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<12}"
        )
        click.echo("-" * 80)

        for row in result:
            click.echo(
                f"{row[0].strftime('%Y-%m-%d'):<12} "
                f"{float(row[1]):<10.2f} "
                f"{float(row[2]):<10.2f} "
                f"{float(row[3]):<10.2f} "
                f"{float(row[4]):<10.2f} "
                f"{int(row[5]):<12}"
            )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("symbol")
@click.option(
    "--start-date",
    "-s",
    help="Start date in YYYY-MM-DD format",
)
@click.option(
    "--end-date",
    "-e",
    help="End date in YYYY-MM-DD format",
)
@click.option(
    "--output",
    "-o",
    default="price_history.png",
    help="Output file path for the chart",
)
@click.option(
    "--volume/--no-volume",
    default=True,
    help="Whether to include volume in the plot",
)
def price_history(symbol, start_date, end_date, output, volume):
    """
    Generate a price history chart for a stock symbol.

    Usage: python cli.py price-history AAPL -s 2023-01-01 -o aapl_chart.png
    """
    try:
        analyzer = StockAnalysis()
        output_path = ensure_extension(output, ".png")

        click.echo(f"Generating price history chart for {symbol}...")
        analyzer.plot_price_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            show_volume=volume,
            save_path=output_path,
        )
        click.echo(f"Chart saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("symbol")
@click.option(
    "--start-date",
    "-s",
    help="Start date in YYYY-MM-DD format",
)
@click.option(
    "--end-date",
    "-e",
    help="End date in YYYY-MM-DD format",
)
@click.option(
    "--output",
    "-o",
    default="dashboard.png",
    help="Output file path for the dashboard",
)
def dashboard(symbol, start_date, end_date, output):
    """
    Generate a comprehensive performance dashboard for a stock symbol.

    Usage: python cli.py dashboard AAPL -s 2022-01-01 -o aapl_dashboard.png
    """
    try:
        analyzer = StockAnalysis()
        output_path = ensure_extension(output, ".png")

        click.echo(f"Generating performance dashboard for {symbol}...")
        analyzer.create_performance_dashboard(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            save_path=output_path,
        )
        click.echo(f"Dashboard saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--start-date",
    "-s",
    help="Start date in YYYY-MM-DD format",
)
@click.option(
    "--end-date",
    "-e",
    help="End date in YYYY-MM-DD format",
)
@click.option(
    "--output",
    "-o",
    default="comparison.png",
    help="Output file path for the comparison chart",
)
@click.option(
    "--normalized/--absolute",
    default=True,
    help="Whether to normalize prices to start at 100 (default) or show absolute prices",
)
def compare(symbols, start_date, end_date, output, normalized):
    """
    Compare performance of multiple stocks.

    Usage: python cli.py compare AAPL MSFT GOOGL -s 2023-01-01 -o comparison.png
    """
    try:
        analyzer = StockAnalysis()
        output_path = ensure_extension(output, ".png")

        if len(symbols) < 2:
            click.echo("Please provide at least two symbols to compare.")
            return

        click.echo(f"Generating comparison chart for {', '.join(symbols)}...")
        analyzer.compare_stocks(
            symbols=list(symbols),
            start_date=start_date,
            end_date=end_date,
            normalized=normalized,
            save_path=output_path,
        )
        click.echo(f"Comparison chart saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("symbol")
@click.option(
    "--start-date",
    "-s",
    help="Start date in YYYY-MM-DD format",
)
@click.option(
    "--end-date",
    "-e",
    help="End date in YYYY-MM-DD format",
)
@click.option(
    "--period",
    "-p",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="daily",
    help="Period for return calculation",
)
@click.option(
    "--output",
    "-o",
    default="returns.png",
    help="Output file path for the returns chart",
)
def returns(symbol, start_date, end_date, period, output):
    """
    Analyze and plot the distribution of stock returns.

    Usage: python cli.py returns AAPL -p monthly -o aapl_returns.png
    """
    try:
        analyzer = StockAnalysis()
        output_path = ensure_extension(output, ".png")

        click.echo(f"Generating {period} returns distribution for {symbol}...")
        analyzer.plot_returns_distribution(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            save_path=output_path,
        )
        click.echo(f"Returns distribution chart saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--table-name",
    "-t",
    default="stock_prices",
    help="Name of the database table to query",
)
def list_symbols(table_name):
    """
    List all stock symbols available in the database.

    Usage: python cli.py list-symbols
    """
    try:
        analyzer = StockAnalysis()
        query = text(f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol ASC")

        with analyzer.engine.connect() as connection:
            result = connection.execute(query).fetchall()

        if not result:
            click.echo("No stock symbols found in the database.")
            return

        symbols = [row[0] for row in result]
        click.echo(f"Found {len(symbols)} stock symbols:")
        for i, symbol in enumerate(symbols):
            click.echo(f"{symbol}", nl=False)
            if i < len(symbols) - 1:
                click.echo(", ", nl=False)
                if (i + 1) % 10 == 0:
                    click.echo()
        click.echo()

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


def ensure_extension(filepath, extension):
    """Ensure file path has the correct extension"""
    if not filepath.lower().endswith(extension):
        filepath += extension

    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    return filepath


if __name__ == "__main__":
    cli()
