import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm, t

## Functions

def multiply_by_100_plot(x, pos):
    """Format function to multiply ticks by 100 for percentages"""
    return f'{x * 100:.1f}'

def process_ticker(ticker, start_date, end_date):
    """Process a single ticker: Calculate filtered data, fit distributions, and compute MAE."""
    # Step 1: Data Retrieval
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close = data['Adj Close']

    # Step 2: Calculate daily returns
    daily_returns = adj_close.pct_change().dropna() * np.sqrt(365.25)
    daily_returns = np.array(daily_returns)

    # Calculate 1st and 99th percentiles
    percentile_1 = np.percentile(daily_returns, 0.5)
    percentile_99 = np.percentile(daily_returns, 99.5)

    # Filter data to include only values within 1st and 99th percentiles
    filtered_data = daily_returns[
        (daily_returns >= percentile_1) & (daily_returns <= percentile_99)
    ]

    # Step 3: Fit Distributions
    distributions = {'Normal': norm, 'Student\'s t': t}
    mape_values = []
    distribution_names = []
    distribution_params = []

    for name, dist in distributions.items():
        try:
            params = dist.fit(daily_returns)

            # Extract parameters
            if name == "Normal":
                norm_mu = params[0]
            elif name == "Student's t":
                t_df = params[0]
                t_mu = params[1]

            bin_centers = 0.5 * (np.histogram(filtered_data, bins=50)[1][:-1] + np.histogram(filtered_data, bins=50)[1][1:])
            fitted_pdf = dist.pdf(bin_centers, *params)
            counts, _ = np.histogram(filtered_data, bins=50, density=True)

            mape = np.mean(np.abs((counts - fitted_pdf)/fitted_pdf)) * 100 # Mean Absolute Error
            mape_values.append(mape)
            distribution_names.append(name)
            distribution_params.append(params)
        except Exception as e:
            print(f"Could not fit distribution {name} for {ticker}: {e}")

    return filtered_data, mape_values, distribution_names, distribution_params, norm_mu, t_mu, t_df

def preprocessing_df(options_df, DTE_min, DTE_max):
    # Step 1: Filtering and column deletion - use more efficient methods
    options_df = options_df[
        (options_df['[DTE]'].between(DTE_min, DTE_max))
    ].drop(columns=['[C_DELTA]', '[P_DELTA]'], errors='ignore')

    # Step 2: Vectorized mapping of expiration prices
    # Create a more efficient mapping of quote dates to underlying last prices
    expire_price_map = options_df.groupby('[QUOTE_DATE]')['[UNDERLYING_LAST]'].first()
    options_df['expire_price'] = options_df['[EXPIRE_DATE]'].map(expire_price_map)

    # Step 3: Vectorized calculations for trade results
    options_df['buy call %'] = (
        np.maximum(options_df['expire_price'] - options_df['[STRIKE]'], 0) - 
        options_df['[C_BID]']
    ) / options_df['[C_BID]'] * 100.0

    options_df['buy put %'] = (
        np.maximum(options_df['[STRIKE]'] - options_df['expire_price'], 0) - 
        options_df['[P_BID]']
    ) / options_df['[P_BID]'] * 100.0

    options_df['sell put %'] = (
        options_df['[P_ASK]']
        - np.maximum(options_df['[STRIKE]'] - options_df['expire_price'], 0)
    ) / (options_df['[STRIKE]'] - options_df['[P_ASK]']) * 100.0

    return options_df

def plot_distributions(ax, ticker, filtered_data, distribution_params, distribution_names):
    """Plot histogram and fitted distributions."""
    ax.hist(filtered_data, bins=50, alpha=0.2, density=True, color='k', edgecolor='black')

    distributions = {'Normal': norm, 'Student\'s t': t}

    for name, params in zip(distribution_names, distribution_params):
        dist = distributions[name]
        x = np.linspace(min(filtered_data), max(filtered_data), 1000)
        ax.plot(x, dist.pdf(x, *params), label=name)

    ax.set_title(ticker)
    ax.set_xlabel('Daily Returns annualized [%]')
    ax.set_ylabel('Density [%]')
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_formatter(FuncFormatter(multiply_by_100_plot))

# Select volatility type and add to options_df
def add_volatility(options_df, IV):

    # Check if IV is a DataFrame
    if isinstance(IV, pd.DataFrame):
        # Rename the column to 'VIX'
        if IV.shape[1] == 1:  # Ensure it's a single-column DataFrame
            IV.columns = ['IV']
        else:
            raise ValueError("The DataFrame 'IV' should only have one column.")
    elif isinstance(IV, pd.Series):
        # Convert Series to DataFrame and rename the column
        IV = IV.to_frame(name='IV')
    else:
        raise TypeError("'IV' must be a pandas DataFrame or Series.")
    IV.index.name = '[QUOTE_DATE]'

    # Ensure the index of IV is datetime if not already
    IV.index = pd.to_datetime(IV.index)
    options_df['[QUOTE_DATE]'] = pd.to_datetime(options_df['[QUOTE_DATE]'])

    # Merge the IV values into options_df based on 'QUOTE_DATE'
    options_df = options_df.merge(IV, left_on='[QUOTE_DATE]', right_index=True, how='left', suffixes=('_delete', ''))
    if 'IV_delete' in options_df.columns:
        del options_df['IV_delete']

    return options_df


def cdf(x, norm_mu, t_mu, t_df, dist_select, sigma):
    """
    Vectorized CDF calculation for normal and t-distributions
    
    Parameters:
    x : array-like, input values
    norm_mu : float, mean for normal distribution
    t_mu : float, mean for t-distribution
    t_df : float, degrees of freedom for t-distribution
    dist_select : str, 'norm' or 't'
    sigma : float or array-like, standard deviation
    
    Returns:
    array of CDF values
    """
    # Convert inputs to numpy arrays for vectorization
    x = np.asarray(x)
    sigma = np.asarray(sigma)
    
    # Preallocate output array
    cdf_values = np.zeros_like(x, dtype=float)
    
    if dist_select == 'norm':
        # Vectorized normalization and CDF for normal distribution
        x_normalized = (x - norm_mu) / sigma
        cdf_values = norm.cdf(x_normalized)
    
    elif dist_select == 't':
        # Vectorized normalization and CDF for t-distribution
        x_normalized = (x - t_mu) / sigma
        cdf_values = t.cdf(x_normalized, t_df)
    
    return cdf_values

def calculate_ev_and_pop(options_df, norm_mu, t_mu, t_df, dist_select):
    """
    Calculate Expected Value and Probability of Profit for options
    
    Parameters:
    options_df : pandas DataFrame with option data
    norm_mu : float, mean for normal distribution
    t_mu : float, mean for t-distribution
    t_df : float, degrees of freedom for t-distribution
    dist_select : str, 'norm' or 't'
    
    Returns:
    DataFrame with additional columns for EV and POP
    """
    # Vectorized input extraction with error handling
    try:
        S = options_df['[UNDERLYING_LAST]'].values
        K = options_df['[STRIKE]'].values
        T = options_df['[DTE]'].values * 1/365
        r = options_df['RFR'].values
        sigma = options_df['IV'].values
        P_buy_call = options_df['[C_BID]'].values
        P_buy_put = options_df['[P_BID]'].values
        P_sell_put = options_df['[P_ASK]'].values
    except KeyError as e:
        raise KeyError(f"Missing required column in options_df: {e}")

    # Compute d1 and d2 using numpy vectorization
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Compute the cdf's
    N_d1 = cdf(d1 * sigma, norm_mu, t_mu, t_df, dist_select, sigma)
    N_d2 = cdf(d2 * sigma, norm_mu, t_mu, t_df, dist_select, sigma)
    N_minus_d1 = 1 - N_d1
    N_minus_d2 = 1 - N_d2
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    result_df = options_df.copy()
    
    # Probability of profit calculations
    result_df['buy call POP%'] = N_d2 * 100.0
    result_df['buy put POP%'] = N_minus_d2 * 100.0
    result_df['sell put POP%'] = N_d2 * 100.0

    
    # Expected value calculations with safety checks
    with np.errstate(divide='ignore', invalid='ignore'):
        call_value = (S * N_d1 - K * np.exp(-r * T) * N_d2)
        put_value = (K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1)

        EV_buy_call = call_value - P_buy_call
        EV_buy_put = put_value - P_buy_put
        EV_sell_put = P_sell_put - put_value

        result_df['buy call EV%'] = np.nan_to_num(
            (EV_buy_call / P_buy_call) * 100.0, 
            nan=0.0, posinf=0.0, neginf=0.0
        )

        result_df['buy put EV%'] = np.nan_to_num(
            (EV_buy_put / P_buy_put) * 100.0, 
            nan=0.0, posinf=0.0, neginf=0.0
        )

        result_df['sell put EV%'] = np.nan_to_num(
            (EV_sell_put / (K - P_sell_put)) * 100.0, 
            nan=0.0, posinf=0.0, neginf=0.0
        )
    
    return result_df

# Simulate the backtest
def simulate_backtest(options_df, required_probability_per, required_EV_per, volume_min):

    # Filtering
    df_buy_call = options_df[(options_df['buy call POP%'] > required_probability_per)\
                             & (options_df['buy call EV%'] > required_EV_per)\
                                & options_df['[C_VOLUME]'] >= volume_min]
    df_buy_put = options_df[(options_df['buy put POP%'] > required_probability_per)\
                            & (options_df['buy put EV%'] > required_EV_per)\
                                & options_df['[P_VOLUME]'] >= volume_min]    
    df_sell_put = options_df[(options_df['sell put POP%'] > required_probability_per)\
                            & (options_df['sell put EV%'] > required_EV_per)\
                                & options_df['[P_VOLUME]'] >= volume_min]

    ## For options buying
    # Calculate number of trades (rows) for each date
    df_buy_call_count = df_buy_call.groupby('[QUOTE_DATE]').size()
    df_buy_put_count = df_buy_put.groupby('[QUOTE_DATE]').size()
    df_sell_put_count = df_sell_put.groupby('[QUOTE_DATE]').size()

    # Calculate the mean of 'buy call %' and 'buy put %' for each date separately
    mean_buy_call_perc = df_buy_call.groupby('[QUOTE_DATE]')['buy call %'].mean()
    mean_buy_put_perc = df_buy_put.groupby('[QUOTE_DATE]')['buy put %'].mean()
    mean_sell_put_perc = df_sell_put.groupby('[QUOTE_DATE]')['sell put %'].mean()

    # Calculate the mean of 'buy call EV%' and 'buy put EV%' for each date separately
    mean_buy_call_ev = df_buy_call.groupby('[QUOTE_DATE]')['buy call EV%'].mean()
    mean_buy_put_ev = df_buy_put.groupby('[QUOTE_DATE]')['buy put EV%'].mean()
    mean_sell_put_ev = df_sell_put.groupby('[QUOTE_DATE]')['sell put EV%'].mean()

    # Create structured dataframes for calls and puts separately
    df_buy_call = pd.DataFrame({
        '[QUOTE_DATE]': mean_buy_call_perc.index,
        'return %': mean_buy_call_perc,
        'EV%': mean_buy_call_ev,
        'N trades': df_buy_call_count
    }).set_index('[QUOTE_DATE]')

    df_buy_put = pd.DataFrame({
        '[QUOTE_DATE]': mean_buy_put_perc.index,
        'return %': mean_buy_put_perc,
        'EV%': mean_buy_put_ev,
        'N trades': df_buy_put_count
    }).set_index('[QUOTE_DATE]')

    df_sell_put = pd.DataFrame({
        '[QUOTE_DATE]': mean_sell_put_perc.index,
        'return %': mean_sell_put_perc,
        'EV%': mean_sell_put_ev,
        'N trades': df_sell_put_count
    }).set_index('[QUOTE_DATE]')

    ## Perform the backtest

    ## Buy call
    df_buy_call['EV% mean'] = df_buy_call['EV%'].expanding(min_periods=10).mean()
    df_buy_call['actual return % mean'] = df_buy_call['return %'].expanding(min_periods=10).mean()
    df_buy_call['N trades total'] = df_buy_call['N trades'].cumsum()
    df_buy_call['MAPE'] = ((df_buy_call['actual return % mean'] - df_buy_call['EV% mean']).abs()/df_buy_call['EV% mean'].abs()) * 100

    ## Buy put
    df_buy_put['EV% mean'] = df_buy_put['EV%'].expanding(min_periods=10).mean()
    df_buy_put['actual return % mean'] = df_buy_put['return %'].expanding(min_periods=10).mean()
    df_buy_put['N trades total'] = df_buy_put['N trades'].cumsum()
    df_buy_put['MAPE'] = ((df_buy_put['actual return % mean'] - df_buy_put['EV% mean']).abs()/df_buy_put['EV% mean'].abs()) * 100

    ## Sell put
    df_sell_put['EV% mean'] = df_sell_put['EV%'].expanding(min_periods=10).mean()
    df_sell_put['actual return % mean'] = df_sell_put['return %'].expanding(min_periods=10).mean()
    df_sell_put['N trades total'] = df_sell_put['N trades'].cumsum()
    df_sell_put['MAPE'] = ((df_sell_put['actual return % mean'] - df_sell_put['EV% mean']).abs()/df_sell_put['EV% mean'].abs()) * 100

    return df_buy_call, df_buy_put, df_sell_put

def combined_backtest(options_df, t_norm, t_mu, t_df, dist_select, required_probability_per, required_EV_per, volume_min):

    options_df = calculate_ev_and_pop(options_df, t_norm, t_mu, t_df, dist_select)
    df_buy_call_mean, df_buy_put_mean, df_sell_put_mean = \
        simulate_backtest(options_df, required_probability_per, required_EV_per, volume_min)

    return df_buy_call_mean, df_buy_put_mean, df_sell_put_mean
