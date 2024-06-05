# %% [markdown]
# <a href="https://colab.research.google.com/github/ThatNinjaGuy/Machine-learning-A-Z-Course/blob/develop/stock_market_strategy_reinforcement_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
!pip install stable-baselines3[extra]

# %%
import re
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import newton
import torch
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, db
import pickle
import glob
from stable_baselines3 import PPO


# %%
class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.holdings = 0

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.float16)

        # Observations
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.cash = self.initial_cash
        self.holdings = 0
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Volume'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['Signal'],
            self.cash + self.holdings * self.df.iloc[self.current_step]['Close']
        ])
        if np.isnan(frame).any() or np.isinf(frame).any():
            print("Invalid observation detected: ", frame)
            raise ValueError("Invalid observation detected")
        return frame

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        reward = self.cash + self.holdings * self.df.iloc[self.current_step]['Close'] - self.initial_cash
        self.reward = reward
        self.total_reward += reward

        if self.done:
            obs = self.reset()  # Reset environment if done
        else:
            obs = self._next_observation()

        return obs, reward, self.done, {}

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        action_type = action[0]

        if action_type < 1:  # Buy
            num_shares_to_buy = self.cash // current_price
            self.cash -= num_shares_to_buy * current_price
            self.holdings += num_shares_to_buy
        elif action_type < 2:  # Sell
            self.cash += self.holdings * current_price
            self.holdings = 0

    def render(self, mode='human', close=False):
        profit = self.cash + self.holdings * self.df.iloc[self.current_step]['Close'] - self.initial_cash
        print(f'Step: {self.current_step}')
        print(f'Cash: {self.cash}')
        print(f'Holdings: {self.holdings}')
        print(f'Profit: {profit}')


# %%
# Function to convert dataframe to dictionary
def df_to_dict(df):
    return df.to_dict(orient='index')

def plot_strategy_performance(data, portfolio_col, strategy_name):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data[portfolio_col], label=portfolio_col)
    plt.title(strategy_name)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


# %%
# Function to initialize Firebase app
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('/Users/deadshot/Desktop/Code/Stock Predictor/stockmarket-b9c19-firebase-adminsdk-uqexr-f4be4c7249.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://stockmarket-b9c19-default-rtdb.firebaseio.com/'
        })

# Function to upload data to firebase in chunks
def upload_data_to_firebase(data, stock_name):
    # Initialize Firebase app if not already initialized
    initialize_firebase()

    # Reference to the Firebase database
    ref = db.reference('stocks/' + stock_name)

    # Reset the index and convert timestamps to strings
    data.reset_index(inplace=True)  # Reset the index to ensure it's unique
    data['Timestamp'] = data['Timestamp'].astype(str)  # Convert Timestamps to strings

    # Replace out-of-range float values with None
    data = data.replace({float('inf'): None, float('-inf'): None, float('nan'): None})

    # Convert the dataframe to a dictionary
    data_dict = df_to_dict(data)

    # Upload data in chunks
    batch_size = 500  # Adjust batch size as needed
    # Uncomment below to upload in chunks. Requires a auth file from firebase
    for i in range(0, len(data_dict), batch_size):
        chunk = dict(list(data_dict.items())[i:i + batch_size])
        ref.update(chunk)

    print("Data uploaded to Firebase successfully.")


# %%

def train_rl_agent(df):
    # Initialize the environment and model
    env = DummyVecEnv([lambda: StockTradingEnv(df.copy())])
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=20000)

    return model

# Function to get time periods for backtesting
def get_time_periods(data, period_years):
    periods = []
    start_date = data.index.min()
    end_date = data.index.max()
    current_start = start_date

    while current_start < end_date:
        current_end = current_start + pd.DateOffset(years=period_years)
        if current_end > end_date:
            current_end = end_date
        periods.append((current_start, current_end))
        current_start = current_end + pd.DateOffset(days=1)

    return periods

# Plot functions
def plot_resampled_data(resampled_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    resampled_data['Close'].plot(ax=ax, title='Stock Prices (Close) - Resampled to 30-Minute Intervals', ylabel='Price')
    plt.show()

def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_moving_averages(data, ma_windows):
    fig, axes = plt.subplots(len(ma_windows), 1, figsize=(12, 4 * len(ma_windows)), sharex=True)

    if len(ma_windows) == 1:
        axes = [axes]

    for i, window in enumerate(ma_windows):
        ma = calculate_moving_average(data, window)
        data['Close'].plot(ax=axes[i], label='Close Price', color='blue')
        ma.plot(ax=axes[i], label=f'MA{window}', color='orange')
        axes[i].set_title(f'Moving Average (MA{window})')
        axes[i].set_ylabel(f'MA{window}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_macd(macd, signal):
    fig, ax = plt.subplots(figsize=(12, 6))
    macd.plot(ax=ax, label='MACD', color='b')
    signal.plot(ax=ax, label='Signal Line', color='r')
    ax.set_title('MACD')
    ax.legend()
    plt.show()

def plot_rsi(rsi):
    fig, ax = plt.subplots(figsize=(12, 6))
    rsi.plot(ax=ax, label='RSI', color='g')
    ax.axhline(30, linestyle='--', alpha=0.5, color='r')
    ax.axhline(70, linestyle='--', alpha=0.5, color='r')
    ax.set_title('RSI')
    ax.legend()
    plt.show()

def plot_volume(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    data['Volume'].plot(ax=ax, title='Trading Volumes', ylabel='Volume')
    plt.show()


# Function to convert dataframe to dictionary
def df_to_dict(df):
    return df.to_dict(orient='index')

def rsi_strategy(data, lower_threshold=30, upper_threshold=70):
    data = data.copy()
    data['Signal_RSI'] = 0
    data.loc[data['RSI'] < lower_threshold, 'Signal_RSI'] = 1  # Buy signal
    data.loc[data['RSI'] > upper_threshold, 'Signal_RSI'] = -1  # Sell signal
    data['Position_RSI'] = data['Signal_RSI'].shift()
    return data

def moving_average_signal(data, window):
    data[f'Signal_MA{window}'] = 0
    data[f'Position_MA{window}'] = 0
    data[f'Signal_MA{window}'] = np.where(data['Close'] > data[f'MA{window}'], 1, -1)
    data[f'Position_MA{window}'] = data[f'Signal_MA{window}'].shift()
    return data

# %%
def save_intermediate_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_intermediate_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


# %%
def visualize_rl_strategy(data):
    rl_portfolio_col = 'Portfolio_RL'
    if rl_portfolio_col in data.columns:
        plot_strategy_performance(data, rl_portfolio_col, 'Reinforcement Learning Strategy')
    else:
        print(f"{rl_portfolio_col} column not found in the DataFrame.")


# %%
def backtest_strategy(data, signal_col, portfolio_col, initial_cash, start, end):
    cash = initial_cash
    holdings = 0
    portfolio_value = []

    start_idx, end_idx = data.index.get_loc(start), data.index.get_loc(end)
    period_data = data.iloc[start_idx:end_idx + 1].copy()  # Ensure a deep copy

    for index, row in period_data.iterrows():
        if row[signal_col] == 1:  # Buy signal
            if cash >= row['Close']:  # Check if there's enough cash to buy
                holdings += cash // row['Close']
                cash %= row['Close']
        elif row[signal_col] == -1:  # Sell signal
            cash += holdings * row['Close']
            holdings = 0
        portfolio_value.append(cash + holdings * row['Close'])

    period_data.loc[:, portfolio_col] = portfolio_value  # Use .loc to avoid SettingWithCopyWarning
    total_gain = portfolio_value[-1] - initial_cash
    return period_data, total_gain


def backtest_moving_average_strategies(data, ma_windows, periods):
    data = data[~data.index.duplicated(keep='first')]

    for window in ma_windows:
        for start, end in periods:
            if start not in data.index:
                start = data.index.asof(start)
            if end not in data.index:
                end = data.index.asof(end)

            period_data = data.loc[start:end]
            data_with_strategy = moving_average_signal(data.copy(), window)
            data_with_strategy, total_gain = backtest_strategy(
                data_with_strategy, f'Signal_MA{window}', f'Portfolio_MA{window}', initial_cash=10000, start=start, end=end
            )
            print(f'Total gain from Close Price and MA{window} strategy: ${total_gain:.2f}')

            suffix = f'_MA{window}'
            all_columns = [f'Signal_MA{window}', f'Position_MA{window}', f'Portfolio_MA{window}']
            new_columns = [col + suffix for col in all_columns]
            data_with_strategy.rename(columns=dict(zip(all_columns, new_columns)), inplace=True)

            columns_to_join = [col for col in new_columns if col not in data.columns]
            data = data.join(data_with_strategy[columns_to_join],how='left')
    return data

def backtest_rsi_strategy(data, periods):
    data = data[~data.index.duplicated(keep='first')]

    for start, end in periods:
        if start not in data.index:
            start = data.index.asof(start)
        if end not in data.index:
            end = data.index.asof(end)

        period_data = data.loc[start:end]
        data_with_strategy = rsi_strategy(period_data.copy(), 30, 70)
        data_with_strategy, total_gain_rsi = backtest_strategy(data_with_strategy, 'Signal_RSI', 'Portfolio_RSI', initial_cash=10000, start=start, end=end)
        print(f'Total gain from RSI strategy: ${total_gain_rsi:.2f}')

        suffix = '_RSI'
        all_columns = ['Signal_RSI', 'Position_RSI', 'Portfolio_RSI']
        new_columns = [col + suffix for col in all_columns]
        data_with_strategy.rename(columns=dict(zip(all_columns, new_columns)), inplace=True)

        columns_to_join = [col for col in new_columns if col not in data.columns]
        data = data.join(data_with_strategy[columns_to_join], how='left')
    return data

def visualize_all_strategies(data, ma_windows):
    for window in ma_windows:
        strategy_col = f'Portfolio_MA{window}_MA{window}'
        if strategy_col in data.columns:
            plot_strategy_performance(data, strategy_col, f'Moving Average {window} Strategy')
    if 'Portfolio_RSI_RSI' in data.columns:
        plot_strategy_performance(data, 'Portfolio_RSI_RSI', 'RSI Strategy')
    if 'Portfolio_RL' in data.columns:
        plot_strategy_performance(data, 'Portfolio_RL', 'Reinforcement Learning Strategy')

def plot_strategy_performance(data, portfolio_col, strategy_name):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data[portfolio_col], label=portfolio_col)
    plt.title(strategy_name)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


# %%
def generate_rl_signals(data, model):
    env = StockTradingEnv(data)
    obs = env.reset()
    signals = []

    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        signals.append(action[0])
        if done:
            obs = env.reset()

    # Ensure signals length matches data length
    while len(signals) < len(data):
        signals.append(signals[-1])
    signals = signals[:len(data)]

    signals_df = pd.DataFrame(signals, index=data.index, columns=['RL_Signal'])
    print("RL Signals Generated:")
    return signals_df


def train_and_backtest_rl_agent(data, periods, stock_name):
    model_filename = '/Users/deadshot/Desktop/Code/Stock Predictor/rl_model'

    # Check if the model already exists
    if os.path.exists(model_filename + ".zip"):
        # Load the model
        rl_model = PPO.load(model_filename)
        print("Loaded RL model from disk.")
    else:
        print("Training Reinforcement Learning Agent...")
        rl_model = train_rl_agent(data)
        # Save the model
        rl_model.save(model_filename)
        print("Trained and saved RL model to disk.")

    print("Generating RL signals...")
    signals_df = generate_rl_signals(data, rl_model)
    data['RL_Signal'] = signals_df['RL_Signal']

    print("Backtesting RL Strategy...")
    for start, end in periods:
        if start not in data.index:
            print(f"Adjusting start timestamp {start}...")
            start = data.index.asof(start)
        if end not in data.index:
            print(f"Adjusting end timestamp {end}...")
            end = data.index.asof(end)

        start_idx = data.index.searchsorted(start)
        end_idx = data.index.searchsorted(end)
        period_data = data.iloc[start_idx:end_idx + 1]

        data_with_rl_strategy, total_gain_rl = backtest_strategy(period_data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=10000, start=start, end=end)
        print(f'Total gain from RL strategy: ${total_gain_rl:.2f}')
        data.loc[period_data.index, 'Portfolio_RL'] = data_with_rl_strategy['Portfolio_RL']

    print("Backtested RL Strategy:")
    print(data[['RL_Signal', 'Portfolio_RL']].tail())  # Display last few rows
    return data


# %%
def xirr(cash_flows):
    def npv(rate):
        return sum([cf / (1 + rate) ** ((date - cash_flows[0][0]).days / 365.0) for date, cf in cash_flows])

    try:
        print("Cash Flows for XIRR Calculation:", cash_flows)  # Debugging
        return newton(npv, 0)
    except RuntimeError as e:
        print(f"XIRR calculation failed to converge: {e}")
        return np.nan


def print_strategy_metrics(data, strategies, initial_cash, stock_name):
    for strategy_name, strategy_col in strategies.items():
        final_value, xirr_value, cagr_value = calculate_strategy_metrics(data, strategy_col, initial_cash, stock_name)
        print(f"Strategy: {strategy_name} for {stock_name}")
        print(f"  Final Portfolio Value: ${final_value:,.2f}")
        if not np.isnan(xirr_value):
            print(f"  XIRR: {xirr_value:.2%}")
        else:
            print("  XIRR: Calculation failed to converge")
        print(f"  CAGR: {cagr_value:.2%}\n")


# %%
def backtest_all_strategies(data, ma_windows, periods, stock_name):
    """Backtests all strategies and returns the modified data."""
    data = backtest_moving_average_strategies(data, ma_windows, periods)
    data = backtest_rsi_strategy(data, periods)
    data = train_and_backtest_rl_agent(data, periods, stock_name)
    return data

def plot_all(data_with_values, resampled_data, ma_windows, all_strategies_data):
    """Handles all plotting activities."""
    plot_resampled_data(resampled_data)
    plot_moving_averages(data_with_values, ma_windows)
    plot_macd(data_with_values['MACD'], data_with_values['Signal'])
    plot_rsi(data_with_values['RSI'])
    plot_volume(data_with_values)
    visualize_all_strategies(all_strategies_data, ma_windows)
    visualize_rl_strategy(all_strategies_data)


# %%
def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    return data

def clean_data(data):
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data

def ensure_unique_index(data):
    data = data[~data.index.duplicated(keep='first')]
    data.index = pd.to_datetime(data.index)
    return data

def resample_data(data):
    data = data[~data.index.duplicated(keep='first')]
    # Ensure the index is a datetime index before resampling
    data.index = pd.to_datetime(data.index)
    resampled_data = data.resample('D').ffill().bfill()  # Forward-fill and then backward-fill
    # Ensure 'Close' column is present after resampling
    assert 'Close' in resampled_data.columns, "Column 'Close' is missing after resampling"
    # Ensure no NaN values in 'Close' column after forward filling
    assert not resampled_data['Close'].isna().any(), "NaN values found in 'Close' column after forward filling"
    return resampled_data

def load_and_combine_data(folder_path, ma_windows):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    combined_data = []
    for file_path in csv_files:
        stock_name = os.path.basename(file_path).replace('.csv', '')
        data_with_indicators, _ = process_stock_file(file_path, stock_name, ma_windows)
        data_with_indicators['Stock'] = stock_name  # Add a column to identify the stock
        combined_data.append(data_with_indicators)
    return pd.concat(combined_data, ignore_index=True)

def train_shared_rl_model(combined_data):
    model_filename = '/Users/deadshot/Desktop/Code/Stock Predictor/shared_rl_model'

    # Check if the model already exists
    if os.path.exists(model_filename + ".zip"):
        # Load the model
        rl_model = PPO.load(model_filename)
        print("Loaded shared RL model from disk.")
    else:
        print("Training Shared Reinforcement Learning Agent...")
        env = StockTradingEnv(combined_data)
        rl_model = PPO('MlpPolicy', env, verbose=1)
        rl_model.learn(total_timesteps=10000)
        # Save the model
        rl_model.save(model_filename)
        print("Trained and saved shared RL model to disk.")

    return rl_model

def train_and_backtest_rl_agent_with_shared_model(data, periods, rl_model, stock_name):
    print(f"Generating RL signals for {stock_name}...")
    signals_df = generate_rl_signals(data, rl_model)
    data['RL_Signal'] = signals_df['RL_Signal']

    print(f"Backtesting RL Strategy for {stock_name}...")
    for start, end in periods:
        if start not in data.index:
            start = data.index.asof(start)
        if end not in data.index:
            end = data.index.asof(end)

        start_idx = data.index.searchsorted(start)
        end_idx = data.index.searchsorted(end)
        period_data = data.iloc[start_idx:end_idx + 1]

        data_with_rl_strategy, total_gain_rl = backtest_strategy(period_data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=10000, start=start, end=end)
        print(f'Total gain from RL strategy for {stock_name}: ${total_gain_rl:.2f}')
        data.loc[period_data.index, 'Portfolio_RL'] = data_with_rl_strategy['Portfolio_RL']

    print(f"Backtested RL Strategy for {stock_name}:")
    print(data[['RL_Signal', 'Portfolio_RL']].tail())  # Display last few rows
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    print("Data type of 'Close' column:", data['Close'].dtype)  # Debugging
    assert data['Close'].dtype in [np.float64, np.float32], "'Close' column must be a float type"

    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    print("MACD and Signal calculated successfully.")  # Debugging
    return macd, signal

def cagr(initial_value, final_value, years):
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return np.nan
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_strategy_metrics(data, strategy_col, initial_cash, stock_name):
    # Filter out NaN values
    valid_data = data.dropna(subset=[strategy_col])

    # If there are no valid data points, return NaN for all metrics
    if valid_data.empty:
        return np.nan, np.nan, np.nan

    final_value = valid_data[strategy_col].iloc[-1]
    total_days = (valid_data.index[-1] - valid_data.index[0]).days
    years = total_days / 365.0

    # XIRR calculation assumes initial investment at the start and final portfolio value at the end
    cash_flows = [(valid_data.index[0], -initial_cash), (valid_data.index[-1], final_value)]
    print(f"Cash Flows for XIRR ({stock_name}): {cash_flows}")

    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    return final_value, xirr_value, cagr_value

def calculate_indicators(data, ma_windows):
    for window in ma_windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()

    data['MACD'], data['Signal'] = calculate_macd(data)
    data['RSI'] = calculate_rsi(data)
    return data

def process_stock_file(file_path, stock_name, ma_windows):
    data = load_data(file_path)
    print(f"Data columns after loading {stock_name}: {data.columns}")

    data = clean_data(data)
    assert not data['Close'].isna().any(), "NaN values found in 'Close' column after data cleaning"

    data = ensure_unique_index(data)
    assert not data['Close'].isna().any(), "NaN values found in 'Close' column after unique indexing"

    resampled_data = resample_data(data)
    assert not resampled_data['Close'].isna().any(), "NaN values found in 'Close' column after forward resample"

    data_with_indicators = calculate_indicators(resampled_data, ma_windows)

    data_with_indicators.sort_index(inplace=True)
    return data_with_indicators, resampled_data


# %%
def final_reinforcement_learning_analysis(rl_model, data, initial_cash):
    print("Running final reinforcement learning analysis on data...")
    signals_df = generate_rl_signals(data, rl_model)
    data['RL_Signal'] = signals_df['RL_Signal']

    # Perform the backtest on the entire dataset
    data, total_gain_rl = backtest_strategy(data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=10000, start=data.index[0], end=data.index[-1])

    print(f'Total gain from final RL strategy: ${total_gain_rl:.2f}')

    # Prepare cash flows for XIRR calculation
    initial_date = data.index[0]
    final_date = data.index[-1]
    final_value = data['Portfolio_RL'].iloc[-1]
    cash_flows = [(initial_date, -initial_cash), (final_date, final_value)]
    print("Cash Flows for final XIRR Calculation:", cash_flows)  # Debugging

    return data, cash_flows, final_value


# %%
def save_performance_metrics(performance_metrics, filename):
    df = pd.DataFrame(performance_metrics)
    df.to_csv(filename, index=False)
    print(f"Performance metrics saved to {filename}")

def save_trade_details(trade_details, stock_name, folder):
    filename = os.path.join(folder, f"{stock_name}_trade_details.csv")
    df = pd.DataFrame(trade_details)
    df.to_csv(filename, index=False)
    print(f"Trade details for {stock_name} saved to {filename}")

def final_reinforcement_learning_analysis(rl_model, stock_data, stock_name, initial_cash, output_folder):
    print(f"Running final reinforcement learning analysis on {stock_name}...")
    signals_df = generate_rl_signals(stock_data, rl_model)
    stock_data['RL_Signal'] = signals_df['RL_Signal']

    # Perform the backtest on the entire stock dataset
    stock_data, total_gain_rl = backtest_strategy(stock_data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=initial_cash, start=stock_data.index[0], end=stock_data.index[-1])

    # Extract trade details
    trade_details = stock_data[['RL_Signal', 'Cash', 'Holdings', 'Portfolio_RL']].dropna().reset_index()

    # Save trade details for the stock
    save_trade_details(trade_details, stock_name, output_folder)

    # Calculate performance metrics
    final_value = stock_data['Portfolio_RL'].iloc[-1]
    total_days = (stock_data.index[-1] - stock_data.index[0]).days
    years = total_days / 365.0

    # Calculate XIRR and CAGR
    cash_flows = [(stock_data.index[0], -initial_cash), (stock_data.index[-1], final_value)]
    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    print(f"Total gain from final RL strategy on {stock_name}: ${total_gain_rl:.2f}")
    return final_value, xirr_value, cagr_value

# %%

def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    return data

def clean_data(data):
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print("Cleaned column names:", data.columns)  # Debugging
    return data

def ensure_unique_index(data):
    data = data[~data.index.duplicated(keep='first')]
    data.index = pd.to_datetime(data.index)
    return data

def resample_data(data):
    data = data[~data.index.duplicated(keep='first')]
    # Ensure the index is a datetime index before resampling
    data.index = pd.to_datetime(data.index)
    resampled_data = data.resample('D').ffill().bfill()  # Forward-fill and then backward-fill

    # Ensure 'Close' column is present after resampling
    assert 'Close' in resampled_data.columns, "Column 'Close' is missing after resampling"
    # Ensure no NaN values in 'Close' column after forward filling
    assert not resampled_data['Close'].isna().any(), "NaN values found in 'Close' column after forward filling"
    return resampled_data

def setup_initial_data(file_path, ma_windows):
    """Loads, cleans, and prepares the initial data."""
    data = load_data(file_path)
    data = clean_data(data)
    data = ensure_unique_index(data)
    resampled_data = resample_data(data)
    data_with_values = add_computed_values(data, ma_windows)
    data_with_values.sort_index(inplace=True)
    return data_with_values, resampled_data

def add_computed_values(data, ma_windows):
    for window in ma_windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    data['MACD'], data['Signal'] = calculate_macd(data['Close'])
    data['RSI'] = calculate_rsi(data['Close'])
    return data

def load_and_combine_data(folder_path, ma_windows):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    combined_data = []
    for file_path in csv_files:
        stock_name = os.path.basename(file_path).replace('.csv', '')
        data_with_indicators, _ = process_stock_file(file_path, stock_name, ma_windows)
        data_with_indicators['Stock'] = stock_name  # Add a column to identify the stock
        combined_data.append(data_with_indicators)
    return pd.concat(combined_data, ignore_index=True)

def train_shared_rl_model(combined_data):
    model_filename = '/Users/deadshot/Desktop/Code/Stock Predictor/shared_rl_model'

    # Check if the model already exists
    if os.path.exists(model_filename + ".zip"):
        # Load the model
        rl_model = PPO.load(model_filename)
        print("Loaded shared RL model from disk.")
    else:
        print("Training Shared Reinforcement Learning Agent...")
        env = StockTradingEnv(combined_data)
        rl_model = PPO('MlpPolicy', env, verbose=1)
        rl_model.learn(total_timesteps=10000)
        # Save the model
        rl_model.save(model_filename)
        print("Trained and saved shared RL model to disk.")

    return rl_model

def generate_rl_signals(data, model):
    env = StockTradingEnv(data)
    obs = env.reset()
    signals = []

    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        signals.append(action[0])
        if done:
            obs = env.reset()

    # Ensure signals length matches data length
    while len(signals) < len(data):
        signals.append(signals[-1])
    signals = signals[:len(data)]

    signals_df = pd.DataFrame(signals, index=data.index, columns=['RL_Signal'])
    print("RL Signals Generated:")
    return signals_df

def backtest_strategy(data, signal_col, portfolio_col, initial_cash, start, end):
    cash = initial_cash
    holdings = 0
    portfolio_value = []

    start_idx, end_idx = data.index.get_loc(start), data.index.get_loc(end)
    period_data = data.iloc[start_idx:end_idx + 1].copy()  # Ensure a deep copy

    for index, row in period_data.iterrows():
        if row[signal_col] == 1:  # Buy signal
            if cash >= row['Close']:  # Check if there's enough cash to buy
                holdings += cash // row['Close']
                cash %= row['Close']
        elif row[signal_col] == -1:  # Sell signal
            cash += holdings * row['Close']
            holdings = 0
        portfolio_value.append(cash + holdings * row['Close'])

    period_data.loc[:, portfolio_col] = portfolio_value  # Use .loc to avoid SettingWithCopyWarning
    total_gain = portfolio_value[-1] - initial_cash
    return period_data, total_gain

def train_and_backtest_rl_agent_with_shared_model(data, periods, rl_model):
    print("Generating RL signals...")
    signals_df = generate_rl_signals(data, rl_model)
    data['RL_Signal'] = signals_df['RL_Signal']

    print("Backtesting RL Strategy...")
    for start, end in periods:
        if start not in data.index:
            print(f"Adjusting start timestamp {start}...")
            start = data.index.asof(start)
        if end not in data.index:
            print(f"Adjusting end timestamp {end}...")
            end = data.index.asof(end)

        start_idx = data.index.searchsorted(start)
        end_idx = data.index.searchsorted(end)
        period_data = data.iloc[start_idx:end_idx + 1]

        data_with_rl_strategy, total_gain_rl = backtest_strategy(period_data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=10000, start=start, end=end)
        print(f'Total gain from RL strategy: ${total_gain_rl:.2f}')
        data.loc[period_data.index, 'Portfolio_RL'] = data_with_rl_strategy['Portfolio_RL']

    print("Backtested RL Strategy:")
    print(data[['RL_Signal', 'Portfolio_RL']].tail())  # Display last few rows
    return data

def final_reinforcement_learning_analysis(rl_model, stock_data, stock_name, initial_cash, output_folder):
    print(f"Running final reinforcement learning analysis on {stock_name}...")
    signals_df = generate_rl_signals(stock_data, rl_model)
    stock_data['RL_Signal'] = signals_df['RL_Signal']

    # Perform the backtest on the entire stock dataset
    stock_data, total_gain_rl = backtest_strategy(stock_data.copy(), 'RL_Signal', 'Portfolio_RL', initial_cash=initial_cash, start=stock_data.index[0], end=stock_data.index[-1])

    # Extract trade details
    trade_details = stock_data[['RL_Signal', 'Cash', 'Holdings', 'Portfolio_RL']].dropna().reset_index()

    # Save trade details for the stock
    save_trade_details(trade_details, stock_name, output_folder)

    # Calculate performance metrics
    final_value = stock_data['Portfolio_RL'].iloc[-1]
    total_days = (stock_data.index[-1] - stock_data.index[0]).days
    years = total_days / 365.0

    # Calculate XIRR and CAGR
    cash_flows = [(stock_data.index[0], -initial_cash), (stock_data.index[-1], final_value)]
    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    print(f"Total gain from final RL strategy on {stock_name}: ${total_gain_rl:.2f}")
    return final_value, xirr_value, cagr_value

def save_performance_metrics(performance_metrics, filename):
    df = pd.DataFrame(performance_metrics)
    df.to_csv(filename, index=False)
    print(f"Performance metrics saved to {filename}")

def save_trade_details(trade_details, stock_name, folder):
    filename = os.path.join(folder, f"{stock_name}_trade_details.csv")
    df = pd.DataFrame(trade_details)
    df.to_csv(filename, index=False)
    print(f"Trade details for {stock_name} saved to {filename}")

def backtest_all_strategies(data, ma_windows, periods, stock_name):
    for window in ma_windows:
        strategy_col = f'Portfolio_MA{window}_MA{window}'
        data = backtest_moving_average_strategy(data, window, periods, strategy_col, stock_name)
    data = backtest_rsi_strategy(data, periods, stock_name)
    return data

def backtest_moving_average_strategy(data, window, periods, strategy_col, stock_name):
    for start, end in periods:
        period_data = data[start:end]
        signals = generate_ma_signals(period_data, window)
        period_data[strategy_col] = signals
        data.update(period_data)
    return data

def backtest_rsi_strategy(data, periods, stock_name):
    for start, end in periods:
        period_data = data[start:end]
        signals = generate_rsi_signals(period_data)
        period_data['Portfolio_RSI_RSI'] = signals
        data.update(period_data)
    return data

# Dummy implementations for signal generation
def generate_ma_signals(data, window):
    signals = np.where(data['Close'] > data[f'MA{window}'], 1, -1)
    return signals

def generate_rsi_signals(data):
    signals = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    return signals

def final_reinforcement_learning_analysis(rl_model, stock_data, stock_name, initial_cash, output_folder):
    print(f"Running final reinforcement learning analysis on {stock_name}...")
    signals_df = generate_rl_signals(stock_data, rl_model)
    stock_data['RL_Signal'] = signals_df['RL_Signal']

    # Initialize columns
    stock_data['Cash'] = initial_cash
    stock_data['Holdings'] = 0
    stock_data['Portfolio_RL'] = initial_cash

    for i in range(len(stock_data)):
        if i == 0:
            continue
        prev_row = stock_data.iloc[i-1]
        curr_row = stock_data.iloc[i]

        # Carry forward previous values
        stock_data.at[curr_row.name, 'Cash'] = prev_row['Cash']
        stock_data.at[curr_row.name, 'Holdings'] = prev_row['Holdings']

        # Buy signal
        if curr_row['RL_Signal'] > 1:
            num_shares_to_buy = stock_data.at[curr_row.name, 'Cash'] // curr_row['Close']
            stock_data.at[curr_row.name, 'Cash'] -= num_shares_to_buy * curr_row['Close']
            stock_data.at[curr_row.name, 'Holdings'] += num_shares_to_buy

        # Sell signal
        elif curr_row['RL_Signal'] < 1:
            stock_data.at[curr_row.name, 'Cash'] += stock_data.at[curr_row.name, 'Holdings'] * curr_row['Close']
            stock_data.at[curr_row.name, 'Holdings'] = 0

        # Update portfolio value
        stock_data.at[curr_row.name, 'Portfolio_RL'] = stock_data.at[curr_row.name, 'Cash'] + stock_data.at[curr_row.name, 'Holdings'] * curr_row['Close']

    # Save trade details for the stock
    trade_details = stock_data[['RL_Signal', 'Cash', 'Holdings', 'Portfolio_RL']].dropna().reset_index()
    save_trade_details(trade_details, stock_name, output_folder)

    # Calculate performance metrics
    final_value = stock_data['Portfolio_RL'].iloc[-1]
    total_days = (stock_data.index[-1] - stock_data.index[0]).days
    years = total_days / 365.0

    # Calculate XIRR and CAGR
    cash_flows = [(stock_data.index[0], -initial_cash), (stock_data.index[-1], final_value)]
    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    print(f"Total gain from final RL strategy on {stock_name}: ${total_gain_rl:.2f}")
    return final_value, xirr_value, cagr_value



# %%
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
])
logger = logging.getLogger()

def xirr(cash_flows):
    def npv(rate):
        return sum([cf / (1 + rate) ** ((date - cash_flows[0][0]).days / 365.0) for date, cf in cash_flows])

    try:
        print("Cash Flows for XIRR Calculation:", cash_flows)  # Debugging
        return newton(npv, 0)
    except RuntimeError as e:
        print(f"XIRR calculation failed to converge: {e}")
        return np.nan

def calculate_strategy_metrics(data, strategy_col, initial_cash, stock_name):
    # Filter out NaN values
    valid_data = data.dropna(subset=[strategy_col])

    # If there are no valid data points, return NaN for all metrics
    if valid_data.empty:
        return np.nan, np.nan, np.nan

    final_value = valid_data[strategy_col].iloc[-1]
    total_days = (valid_data.index[-1] - valid_data.index[0]).days
    years = total_days / 365.0

    # XIRR calculation assumes initial investment at the start and final portfolio value at the end
    cash_flows = [(valid_data.index[0], -initial_cash), (valid_data.index[-1], final_value)]
    print(f"Cash Flows for XIRR ({stock_name}): {cash_flows}")

    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    return final_value, xirr_value, cagr_value

def print_strategy_metrics(data, strategies, initial_cash, stock_name):
    for strategy_name, strategy_col in strategies.items():
        final_value, xirr_value, cagr_value = calculate_strategy_metrics(data, strategy_col, initial_cash, stock_name)
        print(f"Strategy: {strategy_name} for {stock_name}")
        print(f"  Final Portfolio Value: ${final_value:,.2f}")
        if not np.isnan(xirr_value):
            print(f"  XIRR: {xirr_value:.2%}")
        else:
            print("  XIRR: Calculation failed to converge")
        print(f"  CAGR: {cagr_value:.2%}\n")

def backtest_all_strategies(data, ma_windows, periods, stock_name):
    for window in ma_windows:
        strategy_col = f'Portfolio_MA{window}_MA{window}'
        data = backtest_moving_average_strategy(data, window, periods, strategy_col, stock_name)
        if strategy_col not in data.columns:
            print(f"Warning: {strategy_col} not found in DataFrame for {stock_name} after backtesting.")
    data = backtest_rsi_strategy(data, periods, stock_name)
    return data

def calculate_strategy_metrics(data, strategy_col, initial_cash, stock_name):
    if strategy_col not in data.columns:
        print(f"Error: {strategy_col} not found in DataFrame for {stock_name}. Cannot calculate strategy metrics.")
        return np.nan, np.nan, np.nan

    # Filter out NaN values
    valid_data = data.dropna(subset=[strategy_col])

    # If there are no valid data points, return NaN for all metrics
    if valid_data.empty:
        return np.nan, np.nan, np.nan

    final_value = valid_data[strategy_col].iloc[-1]
    total_days = (valid_data.index[-1] - valid_data.index[0]).days
    years = total_days / 365.0

    # XIRR calculation assumes initial investment at the start and final portfolio value at the end
    cash_flows = [(valid_data.index[0], -initial_cash), (valid_data.index[-1], final_value)]
    print(f"Cash Flows for XIRR for {stock_name}: {cash_flows}")

    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    return final_value, xirr_value, cagr_value

def generate_rl_signals(data, model):
    env = StockTradingEnv(data)
    obs = env.reset()
    signals = []

    for i in range(len(data)):
        action, _states = model.predict(obs)
        action_type = action if np.isscalar(action) else action[0]
        signal = 0  # Default to hold
        if action_type < 1:
            signal = 1  # Buy
        elif action_type >= 1:
            signal = -1  # Sell
        obs, reward, done, info = env.step(action)
        signals.append(signal)

        # Ensure observation resets if done
        if done:
            obs = env.reset()

    # Ensure signals length matches data length
    if len(signals) < len(data):
        signals.extend([signals[-1]] * (len(data) - len(signals)))
    elif len(signals) > len(data):
        signals = signals[:len(data)]

    signals_df = pd.DataFrame(signals, index=data.index, columns=['RL_Signal'])
    print("RL Signals Generated:")
    return signals_df

def backtest_strategy(data, signal_col, portfolio_col, initial_cash, start, end):
    cash = initial_cash
    holdings = 0
    portfolio_value = []

    start_idx, end_idx = data.index.get_loc(start), data.index.get_loc(end)
    period_data = data.iloc[start_idx:end_idx + 1].copy()  # Ensure a deep copy

    for index, row in period_data.iterrows():
        if row[signal_col] >= 1.5:  # Buy signal
            if cash >= row['Close']:  # Check if there's enough cash to buy
                holdings += cash // row['Close']
                cash %= row['Close']
        elif row[signal_col] <= 0.5:  # Sell signal
            cash += holdings * row['Close']
            holdings = 0
        portfolio_value.append(cash + holdings * row['Close'])

    period_data.loc[:, portfolio_col] = portfolio_value  # Use .loc to avoid SettingWithCopyWarning
    total_gain = portfolio_value[-1] - initial_cash
    return period_data, total_gain

def final_reinforcement_learning_analysis(rl_model, stock_data, stock_name, initial_cash, output_folder):
    print(f"Running final reinforcement learning analysis on {stock_name}...")
    signals_df = generate_rl_signals(stock_data, rl_model)
    stock_data['RL_Signal'] = signals_df['RL_Signal']

    # Initialize columns
    stock_data['Cash'] = initial_cash
    stock_data['Holdings'] = 0
    stock_data['Portfolio_RL'] = initial_cash

    for i in range(len(stock_data)):
        if i == 0:
            continue
        prev_row = stock_data.iloc[i-1]
        curr_row = stock_data.iloc[i]

        # Carry forward previous values
        stock_data.at[curr_row.name, 'Cash'] = prev_row['Cash']
        stock_data.at[curr_row.name, 'Holdings'] = prev_row['Holdings']

        # Buy signal
        if curr_row['RL_Signal'] == 1:
            num_shares_to_buy = stock_data.at[curr_row.name, 'Cash'] // curr_row['Close']
            stock_data.at[curr_row.name, 'Cash'] -= num_shares_to_buy * curr_row['Close']
            stock_data.at[curr_row.name, 'Holdings'] += num_shares_to_buy

        # Sell signal
        elif curr_row['RL_Signal'] == -1:
            stock_data.at[curr_row.name, 'Cash'] += stock_data.at[curr_row.name, 'Holdings'] * curr_row['Close']
            stock_data.at[curr_row.name, 'Holdings'] = 0

        # Update portfolio value
        stock_data.at[curr_row.name, 'Portfolio_RL'] = stock_data.at[curr_row.name, 'Cash'] + stock_data.at[curr_row.name, 'Holdings'] * curr_row['Close']

    # Save trade details for the stock
    trade_details = stock_data[['RL_Signal', 'Cash', 'Holdings', 'Portfolio_RL']].dropna().reset_index()
    save_trade_details(trade_details, stock_name, output_folder)

    # Calculate performance metrics
    final_value = stock_data['Portfolio_RL'].iloc[-1]
    total_days = (stock_data.index[-1] - stock_data.index[0]).days
    years = total_days / 365.0

    # Calculate XIRR and CAGR
    cash_flows = [(stock_data.index[0], -initial_cash), (stock_data.index[-1], final_value)]
    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    print(f"Total gain from final RL strategy on {stock_name}: ${final_value - initial_cash:.2f}")
    return final_value, xirr_value, cagr_value

def save_trade_details(trade_details, stock_name, output_folder):
    file_path = os.path.join(output_folder, f"{stock_name}_trade_details.csv")
    trade_details.to_csv(file_path, index=False)
    print(f"Trade details for {stock_name} saved to {file_path}")


def generate_rl_signals(data, model):
    env = StockTradingEnv(data)
    obs = env.reset()
    signals = []

    for _ in range(len(data)):
        action, _states = model.predict(obs)
        action_type = action if np.isscalar(action) else action[0]
        signal = 0  # Default to hold
        if action_type == 1:
            signal = 1  # Buy
        elif action_type == -1:
            signal = -1  # Sell
        obs, reward, done, info = env.step(action)
        signals.append(signal)

    # Ensure signals length matches data length
    while len(signals) < len(data):
        signals.append(signals[-1])
    signals = signals[:len(data)]

    signals_df = pd.DataFrame(signals, index=data.index, columns=['RL_Signal'])
    print("RL Signals Generated:")
    return signals_df

def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    return data

def clean_data(data):
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print("Cleaned column names:", data.columns)  # Debugging
    return data

def ensure_unique_index(data):
    data = data[~data.index.duplicated(keep='first')]
    data.index = pd.to_datetime(data.index)
    return data

def resample_data(data):
    data = data[~data.index.duplicated(keep='first')]
    # Ensure the index is a datetime index before resampling
    data.index = pd.to_datetime(data.index)
    resampled_data = data.resample('D').ffill().bfill()  # Forward-fill and then backward-fill

    # Ensure 'Close' column is present after resampling
    assert 'Close' in resampled_data.columns, "Column 'Close' is missing after resampling"
    # Ensure no NaN values in 'Close' column after forward filling
    assert not resampled_data['Close'].isna().any(), "NaN values found in 'Close' column after forward filling"
    return resampled_data


def backtest_strategy(data, signal_col, portfolio_col, initial_cash, start, end):
    cash = initial_cash
    holdings = 0
    portfolio_value = []

    for i, row in data.iterrows():
        if row[signal_col] == 1:  # Buy signal
            if cash >= row['Close']:  # Check if there's enough cash to buy
                holdings += cash // row['Close']
                cash %= row['Close']
        elif row[signal_col] == -1:  # Sell signal
            cash += holdings * row['Close']
            holdings = 0
        portfolio_value.append(cash + holdings * row['Close'])

    data[portfolio_col] = portfolio_value  # Use .loc to avoid SettingWithCopyWarning
    total_gain = portfolio_value[-1] - initial_cash
    return data, total_gain



# %%
class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.holdings = 0

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.float16)

        # Observations
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.cash = self.initial_cash
        self.holdings = 0
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Volume'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['Signal'],
            self.cash + self.holdings * self.df.iloc[self.current_step]['Close']
        ])
        if np.isnan(frame).any() or np.isinf(frame).any():
            print("Invalid observation detected: ", frame)
            raise ValueError("Invalid observation detected")
        return frame

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            self.current_step = len(self.df) - 1  # Ensure the current step is within bounds

        self._take_action(action)
        reward = self.cash + self.holdings * self.df.iloc[self.current_step]['Close'] - self.initial_cash
        self.reward = reward
        self.total_reward += reward
        obs = self._next_observation()

        return obs, reward, self.done, {}


    def _take_action(self, action):
        if self.current_step >= len(self.df):
            print("Current step out of bounds:", self.current_step)
            raise IndexError("Current step out of bounds")

        if 'Close' not in self.df.columns:
            print("Close column missing in data")
            raise KeyError("Close column missing in data")

        current_price = self.df.iloc[self.current_step]['Close']
        action_type = action[0]

        if action_type < 1:  # Buy
            num_shares_to_buy = self.cash // current_price
            self.cash -= num_shares_to_buy * current_price
            self.holdings += num_shares_to_buy
        elif action_type < 2:  # Sell
            self.cash += self.holdings * current_price
            self.holdings = 0

    def render(self, mode='human', close=False):
        profit = self.cash + self.holdings * self.df.iloc[self.current_step]['Close'] - self.initial_cash
        print(f'Step: {self.current_step}')
        print(f'Cash: {self.cash}')
        print(f'Holdings: {self.holdings}')
        print(f'Profit: {profit}')


def generate_rl_signals(data, model):
    env = StockTradingEnv(data)
    obs = env.reset()
    signals = []

    for _ in range(len(data)):
        action, _states = model.predict(obs)
        action_type = action if np.isscalar(action) else action[0]
        signal = 0  # Default to hold
        if action_type < 1:
            signal = 1  # Buy
        elif action_type >= 1:
            signal = -1  # Sell
        obs, reward, done, info = env.step(action)
        signals.append(signal)
        if done:  # Stop if the environment indicates it's done
            break

    # Ensure signals length matches data length
    while len(signals) < len(data):
        signals.append(signals[-1])
    signals = signals[:len(data)]

    signals_df = pd.DataFrame(signals, index=data.index, columns=['RL_Signal'])
    print("RL Signals Generated:")
    return signals_df

def final_reinforcement_learning_analysis(rl_model, stock_data, stock_name, initial_cash, output_folder):
    print(f"Running final reinforcement learning analysis on {stock_name}...")
    signals_df = generate_rl_signals(stock_data, rl_model)
    stock_data['RL_Signal'] = signals_df['RL_Signal']

    # Initialize columns
    stock_data['Cash'] = initial_cash
    stock_data['Holdings'] = 0
    stock_data['Portfolio_RL'] = initial_cash

    cash_flows = [(stock_data.index[0], -initial_cash)]  # Initial investment
    # Loop to process buy/sell signals
    for i in range(len(stock_data)):
        prev_holdings = stock_data['Holdings'].iloc[i - 1] if i > 0 else 0
        prev_cash = stock_data['Cash'].iloc[i - 1] if i > 0 else initial_cash

        print(f"Processing signal for day {i}")
        if stock_data['RL_Signal'].iloc[i] == 1:  # Buy signal
            if prev_cash >= stock_data['Close'].iloc[i]:  # Check if there's enough cash to buy
                print("Ready to buy stocks")
                num_shares_to_buy = prev_cash // stock_data['Close'].iloc[i]
                stock_data.at[stock_data.index[i], 'Cash'] = prev_cash - (num_shares_to_buy * stock_data['Close'].iloc[i])
                stock_data.at[stock_data.index[i], 'Holdings'] = prev_holdings + num_shares_to_buy
                cash_flows.append((stock_data.index[i], -num_shares_to_buy * stock_data['Close'].iloc[i]))
                print(f"Bought {num_shares_to_buy} shares at {stock_data['Close'].iloc[i]}, Cash left: {stock_data['Cash'].iloc[i]}, Holdings: {stock_data['Holdings'].iloc[i]}")
            else:  # Hold signal
                stock_data.at[stock_data.index[i], 'Cash'] = prev_cash
                stock_data.at[stock_data.index[i], 'Holdings'] = prev_holdings
                cash_flows.append((stock_data.index[i], 0))
                print("Already maxed out")
        elif stock_data['RL_Signal'].iloc[i] == -1:  # Sell signal
            print("Ready to sell stocks")
            stock_data.at[stock_data.index[i], 'Cash'] = prev_cash + (prev_holdings * stock_data['Close'].iloc[i])
            stock_data.at[stock_data.index[i], 'Holdings'] = 0
            cash_flows.append((stock_data.index[i], prev_holdings * stock_data['Close'].iloc[i]))
            print(f"Sold all shares at {stock_data['Close'].iloc[i]}, Cash now: {stock_data['Cash'].iloc[i]}, Holdings: {stock_data['Holdings'].iloc[i]}")
        else:  # Hold signal
            stock_data.at[stock_data.index[i], 'Cash'] = prev_cash
            stock_data.at[stock_data.index[i], 'Holdings'] = prev_holdings
            cash_flows.append((stock_data.index[i], 0))
            print("Hold")

        portfolio_val = stock_data['Cash'].iloc[i] + (stock_data['Holdings'].iloc[i] * stock_data['Close'].iloc[i])
        stock_data.at[stock_data.index[i], 'Portfolio_RL'] = portfolio_val

    # Final portfolio value as a cash flow
    final_value = stock_data['Portfolio_RL'].iloc[-1]
    cash_flows.append((stock_data.index[-1], final_value))

    # Save trade details for the stock
    trade_details = stock_data[['RL_Signal', 'Close', 'Cash', 'Holdings', 'Portfolio_RL']].dropna().reset_index()

    save_trade_details(trade_details, stock_name, output_folder)

    # Calculate performance metrics
    total_days = (stock_data.index[-1] - stock_data.index[0]).days
    years = total_days / 365.0

    # Calculate XIRR and CAGR
    print(f"Cash Flows for XIRR Calculation: {cash_flows}")  # Debugging
    xirr_value = xirr(cash_flows)
    cagr_value = cagr(initial_cash, final_value, years)

    print(f"Total gain from final RL strategy on {stock_name}: ${final_value - initial_cash:.2f}")

    # Plotting portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Portfolio_RL'], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(f'Portfolio Value Over Time for {stock_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'{stock_name}_portfolio_value.png'))
    plt.show()

    return final_value, xirr_value, cagr_value

def main():
    folder_path = '/Users/deadshot/Desktop/Code/Stock Predictor/history/'
    output_folder = '/Users/deadshot/Desktop/Code/Stock Predictor/output/'
    os.makedirs(output_folder, exist_ok=True)

    ma_windows = [5, 10, 20, 50, 100, 200]
    initial_cash = 10000

    # Combine data from all files for training
    combined_data = load_and_combine_data(folder_path, ma_windows)
    print("Combined data columns:", combined_data.columns)

    # Training Phase: Train the shared reinforcement learning model
    rl_model = train_shared_rl_model(combined_data)

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    performance_metrics = []

    for file_path in csv_files:
        try:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            stock_name = re.sub(r'[^a-zA-Z]', '_', stock_name)

            data_with_indicators, resampled_data = process_stock_file(file_path, stock_name, ma_windows)

            all_strategies_data = data_with_indicators.copy()
            periods = get_time_periods(all_strategies_data, 5)

            all_strategies_data = backtest_all_strategies(all_strategies_data, ma_windows, periods, stock_name)

            strategies = {
                f'Moving Average {window}': f'Portfolio_MA{window}_MA{window}' for window in ma_windows
            }
            strategies['RSI'] = 'Portfolio_RSI_RSI'
            strategies['Reinforcement Learning'] = 'Portfolio_RL'

            print(f"\nAnalysis for {stock_name}:")
            for strategy_name, strategy_col in strategies.items():
                final_value, xirr_value, cagr_value = calculate_strategy_metrics(all_strategies_data, strategy_col, initial_cash, stock_name)
                print(f"Strategy: {strategy_name}")
                print(f"  Final Portfolio Value: ${final_value:.2f}")
                print(f"  XIRR: {xirr_value:.2%}")
                print(f"  CAGR: {cagr_value:.2%}")

            plot_all(data_with_indicators, resampled_data, ma_windows, all_strategies_data)

            output_file_path = f'/Users/deadshot/Desktop/Code/Stock Predictor/enhanced_data/{stock_name}_with_strategies.csv'
            all_strategies_data.to_csv(output_file_path, index=False)
            print(f"Data with strategies saved to {output_file_path}")

            upload_data_to_firebase(all_strategies_data, stock_name)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Prediction Phase: Use the trained RL model to predict buy/sell signals and analyze performance
    for file_path in csv_files:
        try:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            stock_name = re.sub(r'[^a-zA-Z]', '_', stock_name)
            data_with_indicators, resampled_data = process_stock_file(file_path, stock_name, ma_windows)

            # Final RL analysis for individual stock
            final_value, xirr_value, cagr_value = final_reinforcement_learning_analysis(rl_model, data_with_indicators, stock_name, initial_cash, output_folder)
            performance_metrics.append({
                'Stock': stock_name,
                'Final Portfolio Value': final_value,
                'XIRR': xirr_value,
                'CAGR': cagr_value
            })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save performance metrics
    save_performance_metrics(performance_metrics, os.path.join(output_folder, "performance_metrics.csv"))

if __name__ == "__main__":
    main()




