from sys import float_repr_style
from typing import List, Tuple, Deque, Any
import random
from collections import deque
from enum import Enum
from dataclasses import dataclass


import numpy as np
import pandas as pd
from rich import print


@dataclass(frozen=True)
class Market:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Order:
    balance: float
    net_worth: float
    crypto_bought: float
    crypto_sold: float
    crypto_held: float


@dataclass(frozen=True)
class State:
    market_history: Deque[Market]
    orders_history: Deque[Order]


class Action(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1

    @classmethod
    @property
    def actions(cls) -> List[Enum]:
        return [e for e in cls]

    @classmethod
    @property
    def names(cls) -> List[str]:
        return [e.name for e in cls]

    @classmethod
    @property
    def values(cls) -> List[int]:
        return [e.value for e in cls]


class TradingEnv:
    """"A custom Bitcoin trading environment"""
    def __init__(self, df: pd.DataFrame, initial_balance: float = 1000.0, lookback_window_size: int = 50) -> None:
        """Define action space and state size and other custom parameters"""
        self.df: pd.DataFrame = df.dropna().reset_index()
        self.df_total_steps: int = len(self.df)-1
        self.initial_balance: float = initial_balance
        self.lookback_window_size: int = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history: Deque[Order] = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history: Deque[Market] = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size: Tuple[int, int] = (self.lookback_window_size, 10)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size: int = 0) -> State:
        self.balance: float = self.initial_balance
        self.net_worth: float = self.initial_balance
        self.prev_net_worth: float = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step: int = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step: int = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step: int = self.lookback_window_size
            self.end_step: int = self.df_total_steps
            
        self.current_step: int = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step: int = self.current_step - i
            order: Order = Order(
                balance=self.balance, 
                net_worth=self.net_worth, 
                crypto_bought=self.crypto_bought,
                crypto_sold=self.crypto_sold,
                crypto_held=self.crypto_held)
            self.orders_history.append(order)
            market: Market = Market(
                open=self.df.loc[current_step, 'Open'],
                high=self.df.loc[current_step, 'High'],
                low=self.df.loc[current_step, 'Low'],
                close=self.df.loc[current_step, 'Close'],
                volume=self.df.loc[current_step, 'Volume'])
            self.market_history.append(market)

        state: State = State(market_history=self.market_history, orders_history=self.orders_history)
        return state

    # Get the data points for the given current_step
    def _next_observation(self) -> State:
        market: Market = Market(
            open=self.df.loc[self.current_step, 'Open'],
            high=self.df.loc[self.current_step, 'High'],
            low=self.df.loc[self.current_step, 'Low'],
            close=self.df.loc[self.current_step, 'Close'],
            volume=self.df.loc[self.current_step, 'Volume'])
        self.market_history.append(market)
        obs: State = State(market_history=self.market_history, orders_history=self.orders_history)
        return obs

    # Execute one time step within the environment
    def step(self, action: Action) -> Tuple[State, float, bool]:
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])
        
        if action == Action.HOLD:
            pass 
        elif action == Action.BUY and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
        elif action == Action.SELL and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        order: Order = Order(
            balance=self.balance, 
            net_worth=self.net_worth, 
            crypto_bought=self.crypto_bought,
            crypto_sold=self.crypto_sold,
            crypto_held=self.crypto_held)
        self.orders_history.append(order)

        reward: float = self.net_worth - self.prev_net_worth
        done: bool = self.net_worth <= self.initial_balance/2
        obs: State = self._next_observation()
        
        return obs, reward, done

    # render environment
    def render(self) -> None:
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
