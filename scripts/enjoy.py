import argparse
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enjoy a 2048 trained agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        default="tdl",
        help="RL Algorithm",
        choices=["ql", "tdl"],
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        required=True,
        help="path to a trained agent",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=1,
        help="number of episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random generator seed",
    )
    args = parser.parse_args()
    return args


def make_policy(algo: str, trained_agent: str) -> NTupleNetworkBasePolicy:
    """
    Makes the policy to enjoy.

    :param algo: Name of the algorithm.
    :param trained_agent: Path to a trained agent.
    :return: Policy.
    """
    algo_policy_map = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent)


def play_game(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
) -> dict[str, Any]:
    """
    Plays a 2048 game.

    :param env: Game environment.
    :param policy: Policy to use.
    :return: Info at the end of the game.
    """
    _observation, info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = info["board"]
        action = policy.predict(state=state)
        _observation, _reward, terminated, truncated, info = env.step(action)

    return info


def enjoy() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    env = gym.make(args.env, render_mode="human")
    policy = make_policy(algo=args.algo, trained_agent=args.trained_agent)

    for _ in trange(args.n_episodes, desc="Enjoy"):
        play_game(env=env, policy=policy)

    env.close()


if __name__ == "__main__":
    enjoy()
