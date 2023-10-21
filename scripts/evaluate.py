import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2048 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        choices=["ql", "tdl", "tdl-small"],
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        help="path to a trained agent",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=1000,
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


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_policy(algo: str, trained_agent: str) -> NTupleNetworkBasePolicy:
    algo_policy_map = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent)


def evaluate() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    env = make_env(env_id=args.env)
    if args.algo is not None and args.trained_agent is not None:
        policy = make_policy(algo=args.algo, trained_agent=args.trained_agent)
    else:
        policy = None

    lengths = []
    rewards = []
    max_tiles = []
    total_score = []

    # Run episodes
    for _ in trange(args.n_episodes, desc="Episode"):
        _observation, info = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            if policy is None:
                action = env.action_space.sample()
            else:
                state = info["board"]
                action = policy.predict(state=state)
            _observation, _reward, terminated, truncated, info = env.step(action)

        lengths.extend(info["episode"]["l"])
        rewards.extend(info["episode"]["r"])
        max_tiles.append(info["max"])
        total_score.append(info["total_score"])
        env.reset()

    env.close()

    # Plot results
    plt.style.use("ggplot")

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(lengths)
    axs[0, 0].set_xlabel("Length")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_title("Length")

    axs[0, 1].hist(rewards)
    axs[0, 1].set_xlabel("Reward")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_title("Reward")

    values, counts = np.unique(max_tiles, return_counts=True)
    labels = np.power(2, values, dtype=int)
    order = np.argsort(labels)
    axs[1, 0].bar(labels[order].astype(str), counts[order])
    axs[1, 0].set_xlabel("Max number")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_title("Max number")

    axs[1, 1].hist(total_score)
    axs[1, 1].set_xlabel("Score")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].set_title("Score")

    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    evaluate()
