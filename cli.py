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


@cli.command()
@click.argument("symbol")
@click.option("--start-date", "-s", help="Start date in YYYY-MM-DD format")
@click.option("--end-date", "-e", help="End date in YYYY-MM-DD format")
@click.option("--p", "-p", type=int, default=1, help="AR order parameter")
@click.option("--d", "-d", type=int, default=0, help="Differencing order parameter")
@click.option("--q", "-q", type=int, default=0, help="MA order parameter")
@click.option("--forecast-days", "-f", type=int, default=30, help="Number of days to forecast")
@click.option("--samples", "-n", type=int, default=1000, help="Number of MCMC samples for Stan")
@click.option("--output", "-o", default="arima_forecast.png", help="Output file path for forecast chart")
@click.option("--save-params", "-sp", is_flag=True, help="Save parameter trace plots")
@click.option("--params-output", "-po", default="arima_params.png", help="Output file path for parameter plots")
@click.option("--save-results", "-sr", is_flag=True, help="Save forecast results to database")
@click.option("--evaluate/--no-evaluate", default=False, help="Evaluate multiple ARIMA models")
@click.option("--models", "-m", help="ARIMA models to evaluate, format: 'p,d,q;p,d,q' (e.g., '1,0,0;1,1,1;2,1,0')")
def forecast_arima(symbol, start_date, end_date, p, d, q, forecast_days, samples, output, save_params, params_output, save_results, evaluate, models):
    """
    Run ARIMA forecast on a stock and save results.
    
    Usage examples:
    - Basic forecast: python cli.py forecast-arima AAPL -p 1 -d 0 -q 0 -f 30
    - Evaluate default models: python cli.py forecast-arima GOOG --evaluate
    - Evaluate custom models: python cli.py forecast-arima MSFT --evaluate -m "1,0,0;1,1,0;2,1,1"
    """
    try:
        from library.forecasting import (
            forecast_stock_price_arima,
            evaluate_arima_models,
            plot_parameter_traces,
        )
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta

        analyzer = StockAnalysis()
        output_path = ensure_extension(output, ".png")

        # Initialize database connection for the forecasting function
        db_connection = analyzer.engine

        # Set default date ranges if not provided
        today = datetime.now().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = today

        if start_date is None:
            # Default to 2 years of historical data if not specified
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        click.echo(
            f"Generating ARIMA forecast for {symbol} (from {start_date} to {end_date})..."
        )

        if evaluate:
            # Run model evaluation for multiple ARIMA configurations
            click.echo("Evaluating ARIMA models...")

            # Parse custom models if provided or use default models
            if models:
                try:
                    # Parse from string like "1,0,0;1,1,1;2,1,0"
                    model_list = []
                    for model_str in models.split(';'):
                        p, d, q = map(int, model_str.split(','))
                        model_list.append((p, d, q))
                    
                    click.echo(f"Evaluating custom model set: {model_list}")
                except ValueError:
                    click.echo("Error: Invalid model format. Use format like '1,0,0;1,1,1;2,1,0'")
                    return
            else:
                # Default models if none provided
                model_list = [(1, 0, 0), (1, 0, 1)]
                click.echo(f"Evaluating default model set: {model_list}")

            try:
                results = evaluate_arima_models(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    holdout_days=forecast_days,
                    models=model_list,
                    db_connection=db_connection,
                    n_samples=samples,
                )

                # Get best model info
                best_model = results["best_model"]
                best_p, best_d, best_q = results["best_order"]

                click.echo(f"Best model: ARIMA({best_p},{best_d},{best_q})")
                click.echo(f"RMSE: {best_model['rmse']:.4f}")

                # Use the pre-generated plots instead of creating new ones
                if "plot" in results:
                    # Main forecast plot
                    click.echo("Saving forecast plot...")
                    results["plot"].savefig(output_path)
                    plt.close(results["plot"])
                    click.echo(f"Forecast chart saved to: {output_path}")
                    
                    # If we have metrics plot, save it as well
                    if "metrics_plot" in results:
                        metrics_path = output_path.replace(".png", "_metrics.png")
                        click.echo("Saving model metrics comparison plot...")
                        results["metrics_plot"].savefig(metrics_path)
                        plt.close(results["metrics_plot"])
                        click.echo(f"Metrics comparison saved to: {metrics_path}")
                        
                    # If we have forecast comparison plot, save it as well
                    if "forecast_comparison_plot" in results:
                        comparison_path = output_path.replace(".png", "_comparison.png")
                        click.echo("Saving forecast comparison plot...")
                        results["forecast_comparison_plot"].savefig(comparison_path)
                        plt.close(results["forecast_comparison_plot"])
                        click.echo(f"Forecast comparison saved to: {comparison_path}")
                        
                else:
                    # Fall back to generating a new plot if the pre-generated ones aren't available
                    click.echo("No pre-generated plots found, creating a new one...")
                    
                    # (rest of your existing plotting code here)
                    fig = plt.figure(figsize=(12, 6))
                    # ... existing plotting code ...
                    
                # Save parameter plots if requested (same as before)
                if save_params and "fit" in best_model.get("forecast", {}):
                    params_path = ensure_extension(params_output, ".png")
                    
                    # Generate parameter plots
                    fig = plot_parameter_traces(
                        fit=best_model["forecast"]["fit"],
                        model_type="stan",
                        fig_title=f"ARIMA({best_p},{best_d},{best_q}) Parameters for {symbol}",
                    )
                    
                    if fig:
                        fig.savefig(params_path)
                        plt.close(fig)
                        click.echo(f"Parameter plots saved to: {params_path}")
                    else:
                        click.echo("Could not generate parameter plots (no valid parameters found)")
                        
            except ValueError as e:
                if "No data found" in str(e):
                    click.echo(f"Error: {str(e)}")
                    click.echo("Please fetch the data first using:")
                    click.echo(f"python cli.py fetch {symbol}")
                    return
                raise

        else:
            # Run single ARIMA model with specified parameters
            click.echo(f"Running ARIMA({p},{d},{q}) model...")

            try:
                # Get the forecast
                result = forecast_stock_price_arima(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    p=p,
                    d=d,
                    q=q,
                    forecast_days=forecast_days,
                    db_connection=db_connection,
                    save_to_db=save_results,
                    n_samples=samples,
                )

                # Save the forecast plot (it's already generated in the function)
                plt.savefig(output_path)
                plt.close()

                click.echo(f"Forecast chart saved to: {output_path}")

                # Save parameter plots if requested
                if save_params and "fit" in result:
                    params_path = ensure_extension(params_output, ".png")

                    # Generate parameter plots
                    fig = plot_parameter_traces(
                        fit=result["fit"],
                        model_type="stan",
                        fig_title=f"ARIMA({p},{d},{q}) Parameters for {symbol}",
                    )

                    if fig:
                        fig.savefig(params_path)
                        plt.close(fig)
                        click.echo(f"Parameter plots saved to: {params_path}")
                    else:
                        click.echo(
                            "Could not generate parameter plots (no valid parameters found)"
                        )

                # Print forecast summary
                forecast_mean = result.get("forecast_mean", [])
                if len(forecast_mean) > 0:
                    last_price = result.get("historical_prices", [])[-1]
                    forecast_end = forecast_mean[-1]
                    pct_change = ((forecast_end / last_price) - 1) * 100
                    direction = "up" if pct_change > 0 else "down"

                    click.echo(f"\nForecast Summary for {symbol}:")
                    click.echo(f"Current price: ${last_price:.2f}")
                    click.echo(
                        f"Forecast ({forecast_days} days): ${forecast_end:.2f} ({direction} {abs(pct_change):.2f}%)"
                    )
                    click.echo(
                        f"95% CI: [${result['forecast_lower'][-1]:.2f}, ${result['forecast_upper'][-1]:.2f}]"
                    )

                    if save_results:
                        click.echo("Forecast results saved to database")

            except ValueError as e:
                if "No data found" in str(e):
                    click.echo(f"Error: {str(e)}")
                    click.echo("Please fetch the data first using:")
                    click.echo(f"python cli.py fetch {symbol}")
                    return
                raise

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
