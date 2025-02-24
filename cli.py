import click
from library.data_ingestion import AlphaVantageIngestion
import logging
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Command line interface for stock data ingestion"""


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


if __name__ == "__main__":
    cli()
