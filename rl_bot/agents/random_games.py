import random

import numpy as np
from rich import print

from environments.trading_env import TradingEnv, Action, State


def random_games(env: TradingEnv, train_episodes: int = 50, training_batch_size: int = 500) -> None:
    average_net_worth: float = 0
    for episode in range(train_episodes):
        state: State = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()
            action: Action = random.choice(Action.actions)
            state, reward, done = env.step(action=action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)
