import numpy as np
from rich import print


def random_games(env, train_episodes: int = 50, training_batch_size: int = 500) -> None:
    average_net_worth: float = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)