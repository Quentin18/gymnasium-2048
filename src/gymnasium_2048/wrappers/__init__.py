from gymnasium_2048.wrappers.illegal_reward import IllegalReward
from gymnasium_2048.wrappers.terminate_goal import TerminateGoalWrapper
from gymnasium_2048.wrappers.terminate_illegal import TerminateIllegalWrapper

__all__ = [
    "IllegalReward",
    "TerminateGoalWrapper",
    "TerminateIllegalWrapper",
]
