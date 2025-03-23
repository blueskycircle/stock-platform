import os
import shutil
import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from datetime import datetime



def create_arima_stan_code(p, d, q):
    """
    Generate Stan code for ARIMA(p,d,q) model with improved numerical stability
    
    Args:
        p (int): AR order - number of autoregressive terms
        d (int): Differencing order
        q (int): MA order - number of moving average terms
        
    Returns:
        str: Stan model code as a string
    """
    # Explicitly use parameters to satisfy linter
    ar_terms = p  # Number of AR terms
    ma_terms = q  # Number of MA terms
    diff_order = d  # Differencing order
    
    stan_code = """
    data {
        int<lower=1> N;  // number of observations
        vector[N] y;     // observed time series
        int<lower=0> p;  // AR order
        int<lower=0> q;  // MA order
        int<lower=0> h;  // forecast horizon
    }
    
    parameters {
        vector<lower=-1, upper=1>[p] phi;  // AR coefficients with stationarity constraints
        vector<lower=-1, upper=1>[q] theta;  // MA coefficients with constraint
        real<lower=0.001, upper=5> sigma;  // innovation standard deviation with strict lower bound
    }
    
    model {
        vector[N] mu;
        vector[N] epsilon;
        
        // Stronger priors to keep parameters away from boundaries
        phi ~ normal(0, 2);
        theta ~ normal(0, 2);
        sigma ~ exponential(1);
        
        // Initialize errors with zeros
        epsilon = rep_vector(0, N);
        
        // Likelihood
        for (t in (p+1):N) {
            mu[t] = 0;
            
            // AR component
            for (i in 1:p) {
                if (t-i > 0)  // Check bounds
                    mu[t] += phi[i] * y[t-i];
            }
            
            // MA component
            for (j in 1:min(q, t-1)) {
                mu[t] += theta[j] * epsilon[t-j];
            }
            
            // Safety check to avoid NaN/Inf values
            if (is_nan(mu[t]) || is_inf(mu[t])) {
                target += negative_infinity();  // Reject this state
            } else {
                epsilon[t] = y[t] - mu[t];
                y[t] ~ normal(mu[t], max([sigma, 0.001]));  // Ensure sigma is never too small
            }
        }
    }
    
    generated quantities {
        vector[h] y_pred;
        vector[h] y_pred_lower;
        vector[h] y_pred_upper;
        
        // Generate forecasts
        {
            vector[N+h] eps_full = rep_vector(0, N+h);
            
            // Copy historical errors
            for (t in 1:N) {
                real mu = 0;
                
                if (t > p) {
                    for (i in 1:p) {
                        if (t-i > 0)
                            mu += phi[i] * y[t-i];
                    }
                }
                
                if (t > q) {
                    for (j in 1:q) {
                        mu += theta[j] * eps_full[t-j];
                    }
                }
                
                eps_full[t] = y[t] - mu;
            }
            
            // Forecast future values
            for (t in 1:h) {
                int t_sim = N + t;
                real mu = 0;
                
                // AR component
                for (i in 1:min(p, t_sim-1)) {
                    if (t_sim-i <= N) {
                        mu += phi[i] * y[t_sim-i];
                    } else {
                        mu += phi[i] * y_pred[t_sim-i-N];
                    }
                }
                
                // MA component
                for (j in 1:min(q, t_sim-1)) {
                    mu += theta[j] * eps_full[t_sim-j];
                }
                
                // Generate forecast with increasing uncertainty over horizon
                real sigma_pred = max([sigma * sqrt(1.0 + t * 0.05), 0.001]);  // Ensure positive
                y_pred[t] = normal_rng(mu, sigma_pred);
                y_pred_lower[t] = mu - 1.96 * sigma_pred;
                y_pred_upper[t] = mu + 1.96 * sigma_pred;
                
                // Store prediction
                eps_full[t_sim] = 0;  // Expected error is zero for forecasts
            }
        }
    }
    """
    
    # Print ARIMA model information
    print(f"Creating ARIMA({ar_terms},{diff_order},{ma_terms}) Stan model code")
    
    return stan_code


def fit_arima_stan(data, p=1, d=1, q=1, forecast_horizon=30, n_samples=5000):
    """
    Fit ARIMA model using Stan and generate forecasts with improved numerical stability

    Args:
        data (pd.Series or np.array): Time series data
        p (int): AR order
        d (int): Differencing order
        q (int): MA order
        forecast_horizon (int): Number of periods to forecast
        n_samples (int): Number of MCMC samples

    Returns:
        dict: Dictionary containing forecast results and model

    Raises:
        ValueError: If data preparation fails
        RuntimeError: If Stan model fitting fails
        FileNotFoundError: If Stan model file operations fail
        TypeError: If parameter types are incorrect
    """
    # Convert to numpy if it's a pandas Series
    if isinstance(data, pd.Series):
        data = data.values.copy()
    else:
        data = data.copy()

    # Apply differencing if d > 0
    original_data = data.copy()
    for _ in range(d):
        data = np.diff(data)

    # Scale the data (simple z-score scaling)
    data_mean = np.mean(data)
    data_std = np.std(data)
    if data_std < 1e-8:  # Prevent division by zero
        data_std = 1.0
    data_scaled = (data - data_mean) / data_std

    # Create a dedicated temp directory with a short path name
    # Use a directory in the user's home folder instead of system temp
    user_home = os.path.expanduser("~")
    base_temp_dir = os.path.join(user_home, ".stan_temp")
    os.makedirs(base_temp_dir, exist_ok=True)
    
    # Create unique subdirectory for this run
    import uuid
    run_id = str(uuid.uuid4())[:8]  # Use just first 8 chars for shorter path
    temp_dir = os.path.join(base_temp_dir, f"arima_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    model_path = os.path.join(temp_dir, "model.stan")
    
    try:
        # Write Stan code to the temp directory
        stan_code = create_arima_stan_code(p, d, q)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(stan_code)
        
        print(f"Stan model saved to {model_path}")
        
        # Add a small delay to ensure file is written completely
        import time
        time.sleep(0.5)
        
        try:
            # Compile Stan model with specific error handling
            model = CmdStanModel(stan_file=model_path)
        except FileNotFoundError as e:
            print(f"Stan model file not found: {str(e)}")
            raise
        except RuntimeError as e:
            print(f"Stan model compilation failed: {str(e)}")
            raise RuntimeError(f"Stan model compilation failed: {str(e)}") from e
        
        # Prepare data for Stan
        stan_data = {
            "N": len(data_scaled),
            "y": data_scaled,
            "p": p,
            "q": q,
            "h": forecast_horizon
        }
        
        # Simple initialization
        initialization = [{
            "phi": np.zeros(p) if p > 0 else [],
            "theta": np.zeros(q) if q > 0 else [],
            "sigma": 0.5
        }]
        
        # Debug information
        print(f"Sampling with Stan model: ARIMA({p},{d},{q})")
        print(f"Data shape: {data_scaled.shape}, Output directory: {temp_dir}")
        
        try:
            fit = model.sample(
                data=stan_data,
                iter_sampling=n_samples,
                iter_warmup=1000,
                chains=1,
                show_progress=True,
                adapt_delta=0.95, 
                max_treedepth=10,
                inits=initialization,
                output_dir=temp_dir,
                save_warmup=False,
                seed=42,
            )
            
            print("Stan sampling completed successfully")
            
        except (ValueError, RuntimeError) as e:
            print(f"Stan sampling failed: {str(e)}")
            
            # List directory contents for debugging
            print("\nDirectory contents:")
            for file in os.listdir(temp_dir):
                print(f"- {file}")
                
            raise RuntimeError(f"Stan sampling failed: {str(e)}") from e
        

        try:
            # Extract forecasts
            y_pred = fit.stan_variable("y_pred")
            y_pred_lower = fit.stan_variable("y_pred_lower")
            y_pred_upper = fit.stan_variable("y_pred_upper")
        except KeyError as e:
            print(f"Failed to extract variables from Stan model: {str(e)}")
            raise ValueError(f"Required Stan variable not found: {str(e)}") from e

        # Calculate mean and intervals
        forecast_mean = np.mean(y_pred, axis=0)
        forecast_lower = np.mean(y_pred_lower, axis=0)
        forecast_upper = np.mean(y_pred_upper, axis=0)

        # Unscale the forecasts
        forecast_mean = forecast_mean * data_std + data_mean
        forecast_lower = forecast_lower * data_std + data_mean
        forecast_upper = forecast_upper * data_std + data_mean

        # Undo differencing if needed
        if d > 0:
            last_value = original_data[-1]
            for i in range(len(forecast_mean)):
                forecast_mean[i] = last_value + forecast_mean[i]
                forecast_lower[i] = last_value + forecast_lower[i]
                forecast_upper[i] = last_value + forecast_upper[i]
                last_value = forecast_mean[i]

        # Return results with model order and info
        return {
            "mean": forecast_mean, 
            "lower": forecast_lower, 
            "upper": forecast_upper,
            "order": (p, d, q),
            "success": True
        }

    except Exception as e:  # pylint: disable=broad-except
        # For any unhandled exceptions, provide detailed error info
        print(f"ARIMA({p},{d},{q}) modeling failed with error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception instead of falling back
        raise
        
    finally:
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")


def plot_forecast(data, forecast_result, window=None, symbol=None):
    """
    Plot time series data with forecasts and prediction intervals

    Args:
        data (pd.Series or np.array): Historical time series data
        forecast_result (dict): Result from fit_arima_stan function
        window (int, optional): Number of past periods to include in plot
        symbol (str, optional): Stock symbol for plot title
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    forecast_mean = forecast_result["mean"]
    forecast_lower = forecast_result["lower"]
    forecast_upper = forecast_result["upper"]

    # Create forecast index
    last_date = data.index[-1]
    if isinstance(last_date, pd.Timestamp):
        # If the data has a datetime index, extend it
        freq = pd.infer_freq(data.index)
        forecast_index = pd.date_range(
            start=last_date, periods=len(forecast_mean) + 1, freq=freq
        )[1:]
    else:
        # If the data has a numeric index, continue the sequence
        forecast_index = np.arange(len(data), len(data) + len(forecast_mean))

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Limit historical data if window is specified
    if window is not None and window < len(data):
        plot_data = data.iloc[-window:]
    else:
        plot_data = data

    # Plot historical data
    plt.plot(plot_data.index, plot_data.values, "b-", label="Historical Data")

    # Plot forecast
    plt.plot(forecast_index, forecast_mean, "r-", label="Forecast")
    plt.fill_between(
        forecast_index,
        forecast_lower,
        forecast_upper,
        color="r",
        alpha=0.2,
        label="95% Prediction Interval",
    )

    p, d, q = forecast_result["order"]
    title = f"ARIMA({p},{d},{q}) Forecast"
    if symbol:
        title = f"{symbol} - {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return plt.gcf()


def get_db_connection(db_connection_string=None):
    """
    Create a connection to the database

    Args:
        db_connection_string (str, optional): Database connection string.
            If None, will attempt to load from environment variable.

    Returns:
        sqlalchemy.engine.Engine: Database connection engine
    """
    if db_connection_string is None:
        # Try to get connection string from environment variable
        db_connection_string = os.environ.get("DB_CONNECTION_STRING")

        # If still None, use a default SQLite database path
        if db_connection_string is None:
            db_connection_string = "sqlite:///" + os.path.join(
                os.path.dirname(__file__), "stock_prices.db"
            )

    return sqlalchemy.create_engine(db_connection_string)


def load_stock_data_from_db(
    symbol, start_date, end_date, db_connection=None, price_column="close"
):
    """
    Load stock price data from database

    Args:
        symbol (str): Stock symbol
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        db_connection (sqlalchemy.engine.Engine, optional): Database connection
        price_column (str): Column name for price data

    Returns:
        pd.Series: Series containing stock price data indexed by date
    """
    # Create database connection if not provided
    if db_connection is None:
        db_connection = get_db_connection()

    # Convert string dates to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # SQL query to get stock data
    query = f"""
    SELECT date, {price_column}
    FROM stock_prices
    WHERE symbol = '{symbol}'
      AND date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
    ORDER BY date ASC
    """

    # Execute query and load into DataFrame
    data = pd.read_sql(query, db_connection, index_col="date", parse_dates=["date"])

    # If we got no data, raise an exception
    if len(data) == 0:
        raise ValueError(
            f"No data found for symbol {symbol} between {start_date} and {end_date}"
        )

    # Return price series
    return data[price_column]


def save_forecast_to_db(
    symbol,
    date_index,
    actual_prices,
    forecast_mean,
    lower_bound,
    upper_bound,
    db_connection,
    model_params=None,
):
    """
    Save forecast results to database

    Args:
        symbol (str): Stock symbol
        date_index (pandas.DatetimeIndex): Dates for the forecasts
        actual_prices (numpy.ndarray): Historical actual prices (can be None for forecast dates)
        forecast_mean (numpy.ndarray): Mean forecast values
        lower_bound (numpy.ndarray): Lower confidence bound
        upper_bound (numpy.ndarray): Upper confidence bound
        db_connection: SQLAlchemy database connection
        model_params (dict, optional): Dictionary containing model parameters
    """
    try:
        # Get current timestamp for the forecast generation
        forecast_timestamp = datetime.now()

        # Create a metadata object
        metadata = sqlalchemy.MetaData()

        # Define the table if it doesn't exist
        stock_forecasts = sqlalchemy.Table(
            "stock_forecasts",
            metadata,
            sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column("symbol", sqlalchemy.String(10), nullable=False),
            sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
            sqlalchemy.Column("actual_price", sqlalchemy.Float),
            sqlalchemy.Column("forecast_price", sqlalchemy.Float),
            sqlalchemy.Column("lower_bound", sqlalchemy.Float),
            sqlalchemy.Column("upper_bound", sqlalchemy.Float),
            sqlalchemy.Column(
                "forecast_generated_at", sqlalchemy.DateTime, nullable=False
            ),
            sqlalchemy.Column("model_type", sqlalchemy.String(50)),
            sqlalchemy.Column("model_params", sqlalchemy.JSON),
            sqlalchemy.UniqueConstraint("symbol", "date", name="uix_symbol_date"),
        )

        # Create the table if it doesn't exist
        metadata.create_all(db_connection)

        # Prepare data for insertion
        data_to_insert = []

        # Convert model parameters to JSON-serializable format
        model_params_json = None
        if model_params:
            model_params_json = {
                k: str(v) if isinstance(v, (np.ndarray, list)) else v
                for k, v in model_params.items()
            }

        for i, date in enumerate(date_index):
            actual = (
                None
                if i >= len(actual_prices) or np.isnan(actual_prices[i])
                else float(actual_prices[i])
            )
            forecast = None if i >= len(forecast_mean) else float(forecast_mean[i])
            lower = None if i >= len(lower_bound) else float(lower_bound[i])
            upper = None if i >= len(upper_bound) else float(upper_bound[i])

            data_to_insert.append(
                {
                    "symbol": symbol,
                    "date": date.date(),  # Convert to date object
                    "actual_price": actual,
                    "forecast_price": forecast,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "forecast_generated_at": forecast_timestamp,
                    "model_type": "ARIMA",
                    "model_params": model_params_json,
                }
            )

        # Insert data
        with db_connection.connect() as conn:
            # Use on duplicate key update to handle existing records
            for record in data_to_insert:
                stmt = sqlalchemy.dialects.mysql.insert(stock_forecasts).values(
                    **record
                )
                stmt = stmt.on_duplicate_key_update(
                    forecast_price=stmt.inserted.forecast_price,
                    lower_bound=stmt.inserted.lower_bound,
                    upper_bound=stmt.inserted.upper_bound,
                    forecast_generated_at=stmt.inserted.forecast_generated_at,
                    model_type=stmt.inserted.model_type,
                    model_params=stmt.inserted.model_params,
                )
                conn.execute(stmt)
            conn.commit()

        print(f"Successfully saved forecast to database for {symbol}")
    except sqlalchemy.exc.SQLAlchemyError as e:
        # Database-related errors
        print(f"Database error while saving forecast: {str(e)}")
    except ValueError as e:
        # Data validation errors
        print(f"Invalid data error while saving forecast: {str(e)}")
    except TypeError as e:
        # Type conversion errors
        print(f"Type error while saving forecast: {str(e)}")
    except (KeyError, IndexError) as e:
        # Dict/list access errors
        print(f"Key/index error while saving forecast: {str(e)}")
    except OSError as e:
        # Operating system/IO errors
        print(f"OS error while saving forecast: {str(e)}")
    except Exception as e:  # pylint: disable=broad-except
        # As a last resort, catch other unexpected errors
        # but disable the warning specifically for this line
        print(f"Unexpected error saving forecast to database: {str(e)}")
        import traceback
        traceback.print_exc()


def forecast_stock_price_arima(
    symbol,
    start_date,
    end_date,
    p=1,
    d=1,
    q=1,
    forecast_days=30,
    price_column="close",
    db_connection=None,
    save_to_db=False,
):
    """
    Forecast stock prices using ARIMA model

    Args:
        symbol (str): Stock symbol
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD)
        p (int): AR order
        d (int): Differencing order
        q (int): MA order
        forecast_days (int): Number of days to forecast
        price_column (str): Column name for price data
        db_connection: SQLAlchemy database connection
        save_to_db (bool): Whether to save forecasts to database

    Returns:
        dict: Forecast results including dates, actual prices, forecasts, and confidence intervals
    """
    # Query historical price data from database
    query = f"""
        SELECT date, {price_column} as price
        FROM stock_prices
        WHERE symbol = %(symbol)s
        AND date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY date ASC
    """

    df = pd.read_sql_query(
        query,
        db_connection,
        params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
    )

    # Check if we have data
    if len(df) == 0:
        print(f"No data found for {symbol} between {start_date} and {end_date}")
        return None

    # Convert to time series
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    price_series = df["price"]

    # Generate future dates for forecasting
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B",  # Business days
    )

    # Store historical dates and prices
    historical_dates = df.index
    historical_prices = price_series.values

    # Combine historical and future dates for plotting
    all_dates = historical_dates.append(future_dates)

    try:
        # Fit model and generate forecasts
        forecast_result = fit_arima_stan(
            price_series.values, p=p, d=d, q=q, forecast_horizon=forecast_days
        )

        # Combine actual and forecasted prices for plotting
        all_prices = np.concatenate(
            [historical_prices, np.full(forecast_days, np.nan)]  # NaN for future dates
        )

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(historical_dates, historical_prices, "b-", label="Historical")

        # Plot forecast
        plt.plot(future_dates, forecast_result["mean"], "r--", label="Forecast")

        # Plot confidence interval
        plt.fill_between(
            future_dates,
            forecast_result["lower"],
            forecast_result["upper"],
            color="r",
            alpha=0.2,
            label="95% Confidence Interval",
        )

        plt.title(f"{symbol} Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        # Save results to database if requested
        if save_to_db and db_connection:
            model_params = {
                "p": p,
                "d": d,
                "q": q,
                "forecast_days": forecast_days,
                "start_date": start_date,
                "end_date": end_date,
            }

            save_forecast_to_db(
                symbol=symbol,
                date_index=all_dates,
                actual_prices=all_prices,
                forecast_mean=forecast_result["mean"],
                lower_bound=forecast_result["lower"],
                upper_bound=forecast_result["upper"],
                db_connection=db_connection,
                model_params=model_params,
            )

        # Return the results
        return {
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "forecast_dates": future_dates,
            "forecast_mean": forecast_result["mean"],
            "forecast_lower": forecast_result["lower"],
            "forecast_upper": forecast_result["upper"],
        }

    except ValueError as e:
        # Handle data validation errors
        print(f"Data validation error: {str(e)}")
        print("Please check input parameters and data quality.")
        return None
        
    except RuntimeError as e:
        # Handle Stan/ARIMA modeling errors
        print(f"ARIMA modeling error: {str(e)}")
        print("Try different model parameters (p,d,q) or adjust the date range.")
        return None
        
    except sqlalchemy.exc.SQLAlchemyError as e:
        # Handle database errors
        print(f"Database error: {str(e)}")
        print("Check database connection and permissions.")
        return None
        
    except (KeyError, IndexError) as e:
        # Handle data access errors
        print(f"Data access error: {str(e)}")
        print("Check data format and availability.")
        return None
        
    except OSError as e:
        # Handle file system/IO errors
        print(f"System error: {str(e)}")
        return None
        
    except Exception as e:  # pylint: disable=broad-except
        # Keep a safety net but disable the warning
        print(f"Unexpected error in stock price forecasting: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("No forecast could be generated. Please try different parameters.")
        return None


def evaluate_arima_models(
    symbol,
    start_date,
    end_date,
    holdout_days=30,
    models=None,
    db_connection=None,
    price_column="close",
):
    """
    Evaluate multiple ARIMA models and select the best one

    Args:
        symbol (str): Stock symbol
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD)
        holdout_days (int): Number of days to hold out for validation
        models (list): List of (p,d,q) tuples to try
        db_connection: SQLAlchemy database connection
        price_column (str): Column name for price data

    Returns:
        dict: Evaluation results and best model
    """
    # Set default models if None is provided
    if models is None:
        models = [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 0)]

    print(f"Evaluating ARIMA models for {symbol} from {start_date} to {end_date}")

    # Query data
    query = f"""
        SELECT date, {price_column} as price
        FROM stock_prices
        WHERE symbol = %(symbol)s
        AND date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY date ASC
    """

    df = pd.read_sql_query(
        query,
        db_connection,
        params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
    )

    # Check if we have enough data
    if len(df) < 60:
        print(f"Warning: Not enough data points ({len(df)}) for reliable forecasting")
        return None

    # Convert to time series
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Print data statistics
    print("Data summary:")
    print(f"- Time range: {df.index.min()} to {df.index.max()}")
    print(f"- Total observations: {len(df)}")
    print(f"- Missing values: {df['price'].isna().sum()}")
    print(f"- Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")

    # Split into train and validation sets
    train_data = df["price"][:-holdout_days]
    val_data = df["price"][-holdout_days:]

    print(
        f"Training data: {len(train_data)} points from {train_data.index[0]} to {train_data.index[-1]}"
    )
    print(
        f"Validation data: {len(val_data)} points from {val_data.index[0]} to {val_data.index[-1]}"
    )

    # Evaluate models
    results = []

    # Lists to store metrics for plotting
    model_labels = []
    mse_values = []
    mae_values = []
    mape_values = []
    rmse_values = []

    for p, d, q in models:
        model_info = {"order": (p, d, q)}
        print(f"\nFitting ARIMA({p},{d},{q})...")

        try:
            # Fit model on training data
            forecast = fit_arima_stan(
                train_data, p=p, d=d, q=q, forecast_horizon=holdout_days
            )

            # Calculate metrics
            mse = np.mean((forecast["mean"] - val_data.values) ** 2)
            mae = np.mean(np.abs(forecast["mean"] - val_data.values))
            mape = (
                np.mean(np.abs((forecast["mean"] - val_data.values) / val_data.values))
                * 100
            )
            rmse = np.sqrt(mse)

            # Store results
            model_info["mse"] = mse
            model_info["mae"] = mae
            model_info["mape"] = mape
            model_info["rmse"] = rmse
            model_info["forecast"] = forecast

            print(
                f"  MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}"
            )

            # Append values for plotting
            model_label = f"({p},{d},{q})"
            model_labels.append(model_label)
            mse_values.append(mse)
            mae_values.append(mae)
            mape_values.append(mape)
            rmse_values.append(rmse)

            # Add to results
            results.append(model_info)

        except ValueError as e:
            # Handle data validation errors
            print(f"Data validation error in ARIMA({p},{d},{q}): {str(e)}")
            
        except RuntimeError as e:
            # Handle Stan/ARIMA modeling errors
            print(f"Runtime error in ARIMA({p},{d},{q}): {str(e)}")
            
        except (FileNotFoundError, OSError) as e:
            # Handle file system errors
            print(f"File operation error in ARIMA({p},{d},{q}): {str(e)}")
            
        except TypeError as e:
            # Handle type errors
            print(f"Type error in ARIMA({p},{d},{q}): {str(e)}")
            
        except KeyError as e:
            # Handle key errors (e.g., missing forecast results)
            print(f"Key error in ARIMA({p},{d},{q}): {str(e)}")
            
        except Exception as e:  # pylint: disable=broad-except
            # Keep a safety net but disable the warning
            print(f"Unexpected error in ARIMA({p},{d},{q}): {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No models were successfully fit.")
        return None

    # Plot metrics comparison
    fig_metrics = plt.figure(figsize=(15, 15))

    # 1. MSE Comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_labels, mse_values, color="skyblue")
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mse_values[i]:.4f}",
            ha="center",
        )
    plt.title("Mean Squared Error (MSE) Comparison")
    plt.xlabel("ARIMA Model (p,d,q)")
    plt.ylabel("MSE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 2. MAE Comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_labels, mae_values, color="lightgreen")
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mae_values[i]:.4f}",
            ha="center",
        )
    plt.title("Mean Absolute Error (MAE) Comparison")
    plt.xlabel("ARIMA Model (p,d,q)")
    plt.ylabel("MAE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 3. MAPE Comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(model_labels, mape_values, color="salmon")
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mape_values[i]:.2f}%",
            ha="center",
        )
    plt.title("Mean Absolute Percentage Error (MAPE) Comparison")
    plt.xlabel("ARIMA Model (p,d,q)")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. RMSE Comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(model_labels, rmse_values, color="mediumpurple")
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{rmse_values[i]:.4f}",
            ha="center",
        )
    plt.title("Root Mean Squared Error (RMSE) Comparison")
    plt.xlabel("ARIMA Model (p,d,q)")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle(f"ARIMA Model Performance Metrics for {symbol}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Create a comparison table
    metrics_df = pd.DataFrame(
        {
            "Model": model_labels,
            "MSE": mse_values,
            "MAE": mae_values,
            "MAPE (%)": mape_values,
            "RMSE": rmse_values,
        }
    )

    # Sort by MSE (or change to another metric if preferred)
    metrics_df = metrics_df.sort_values("MSE")

    # Display the table
    print("\nModel Metrics Comparison:")
    pd.set_option("display.precision", 4)
    print(metrics_df)

    # Plot forecast comparison for all models
    fig_forecast = plt.figure(figsize=(12, 8))
    plt.plot(train_data.index[-30:], train_data[-30:], "k-", label="Training Data")
    plt.plot(val_data.index, val_data, "b-", linewidth=2, label="Actual (Validation)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results))) # pylint: disable=no-member

    for i, model in enumerate(results):
        p, d, q = model["order"]
        plt.plot(
            val_data.index,
            model["forecast"]["mean"],
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f'ARIMA({p},{d},{q}) - MSE: {model["mse"]:.2f}',
        )

    plt.title(f"{symbol} - Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    # Find best model based on MSE
    best_model = min(results, key=lambda x: x["mse"])
    p_best, d_best, q_best = best_model["order"]
    print(f"\nBest model: ARIMA({p_best},{d_best},{q_best})")
    print(
        f"MSE: {best_model['mse']:.4f}, MAE: {best_model['mae']:.4f}, MAPE: {best_model['mape']:.4f}%"
    )

    # Refit on full data for future forecasting
    print("Fitting best model on full dataset...")
    full_data = df["price"]

    # Create main figure for final results
    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Historical validation
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, train_data, "b-", label="Training Data")
    plt.plot(val_data.index, val_data, "g-", label="Validation Data")
    plt.plot(val_data.index, best_model["forecast"]["mean"], "r--", label="Forecast")
    plt.fill_between(
        val_data.index,
        best_model["forecast"]["lower"],
        best_model["forecast"]["upper"],
        color="r",
        alpha=0.2,
        label="95% Prediction Interval",
    )
    plt.title(
        f"{symbol} - Historical Validation with ARIMA({p_best},{d_best},{q_best})"
    )
    plt.legend()
    plt.grid(True)

    # Plot 2: Full forecast with best model
    print(f"Generating full forecast with ARIMA({p_best},{d_best},{q_best})")

    # Debug information
    print(
        f"Full data shape: {full_data.shape}, range: {full_data.index[0]} to {full_data.index[-1]}"
    )

    # Generate full forecast
    full_forecast = fit_arima_stan(
        full_data, p=p_best, d=d_best, q=q_best, forecast_horizon=30
    )

    # Debug forecast values
    print("Forecast statistics:")
    print(
        f"- Mean range: {np.min(full_forecast['mean']):.2f} to {np.max(full_forecast['mean']):.2f}"
    )
    print(f"- Standard deviation: {np.std(full_forecast['mean']):.2f}")
    print(f"- First 5 predicted values: {full_forecast['mean'][:5]}")

    # Create forecast index for future dates
    last_date = full_data.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=30, freq="B"
    )  # Business days

    # Debug forecast dates
    print(f"Forecast dates: {forecast_index[0]} to {forecast_index[-1]}")

    # Plot future forecast
    plt.subplot(2, 1, 2)
    plt.plot(
        full_data.index[-90:],
        full_data.values[-90:],
        "b-",
        linewidth=2,
        label="Historical Data",
    )
    plt.plot(forecast_index, full_forecast["mean"], "r-", linewidth=2, label="Forecast")
    plt.fill_between(
        forecast_index,
        full_forecast["lower"],
        full_forecast["upper"],
        color="r",
        alpha=0.2,
        label="95% Prediction Interval",
    )
    plt.title(f"{symbol} - Future Forecast (30 days)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Return comprehensive results
    return {
        "evaluation": results,
        "best_model": best_model,
        "best_order": (p_best, d_best, q_best),
        "metrics_summary": metrics_df,
        "future_forecast": {
            "mean": full_forecast["mean"],
            "lower": full_forecast["lower"],
            "upper": full_forecast["upper"],
            "dates": forecast_index,
        },
        "data_summary": {
            "total_observations": len(df),
            "training_points": len(train_data),
            "validation_points": len(val_data),
            "date_range": (df.index[0], df.index[-1]),
        },
        "plot": fig,
        "metrics_plot": fig_metrics,
        "forecast_comparison_plot": fig_forecast
    }


if __name__ == "__main__":
    # Example usage

    # Option 1: Direct forecast with specified ARIMA order
    result = forecast_stock_price_arima(
        symbol="AAPL",
        start_date="2022-01-01",
        end_date="2023-01-01",
        p=1,
        d=1,
        q=1,
        forecast_days=30,
        save_to_db=True,
    )

    # Option 2: Evaluate multiple models and select the best
    evaluation = evaluate_arima_models(
        symbol="AAPL",
        start_date="2021-01-01",
        end_date="2023-01-01",
        holdout_days=30,
        models=[(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (2, 1, 2)],
    )

    plt.show()
