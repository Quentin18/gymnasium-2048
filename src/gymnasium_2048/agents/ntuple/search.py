import numpy as np

from gymnasium_2048.agents.ntuple.policy import NTupleNetworkBasePolicy
from gymnasium_2048.envs import TwentyFortyEightEnv


class ExpectimaxSearch:
    def __init__(
        self,
        policy: NTupleNetworkBasePolicy,
        max_depth: int = 3,
    ) -> None:
        self.policy = policy
        self.max_depth = max_depth
        self.min_value = 0.0

    def _evaluate(self, state: np.ndarray) -> tuple[float, int]:
        values = [
            self.policy.evaluate(state=state, action=action) for action in range(4)
        ]
        max_action = np.argmax(values)
        return max(self.min_value, values[max_action]), max_action

    def _maximize(self, state: np.ndarray, depth: int) -> tuple[float, int]:
        if depth >= self.max_depth:
            return self._evaluate(state=state)

        max_value = self.min_value
        max_action = 0

        for action in range(4):
            after_state, _, is_legal = TwentyFortyEightEnv.apply_action(
                board=state,
                action=action,
            )
            if not is_legal:
                continue

            value = self._chance(after_state=after_state, depth=depth + 1)
            if value > max_value:
                max_value = value
                max_action = action

        return max_value, max_action

    def _chance(self, after_state: np.ndarray, depth: int) -> float:
        if depth >= self.max_depth:
            return self._evaluate(state=after_state)[0]

        values, weights = [], []

        for row in range(after_state.shape[0]):
            for col in range(after_state.shape[1]):
                if after_state[row, col] != 0:
                    continue

                for value, prob in ((1, 0.9), (2, 0.1)):
                    after_state[row, col] = value
                    values.append(self._maximize(state=after_state, depth=depth + 1)[0])
                    weights.append(prob)
                    after_state[row, col] = 0

        return np.average(values, weights=weights)

    def predict(self, state: np.ndarray) -> int:
        value, action = self._maximize(state=state, depth=0)
        print(value, action)
        return action
