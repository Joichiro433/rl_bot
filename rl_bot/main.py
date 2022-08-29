import pandas as pd
import pretty_errors

from environments.trading_env import TradingEnv
from agents.random_games import random_games
import params


if __name__ == '__main__':
    df = pd.read_csv(params.DATA_PATH)
    df = df.sort_values('Date')

    lookback_window_size = 10
    train_df = df[:-720-lookback_window_size]
    test_df = df[-720-lookback_window_size:] # 30 days

    train_env = TradingEnv(df=train_df, lookback_window_size=lookback_window_size)
    test_env = TradingEnv(df=test_df, lookback_window_size=lookback_window_size)   

    random_games(train_env, train_episodes = 10, training_batch_size=500)
