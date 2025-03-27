from abc import ABC, abstractmethod
from typing import Sequence, Any
import numpy as np
import numpy.typing as npt


class Mechanism(ABC):
    def __init__(self, agent_budget_per_collector_step: float = 10.0):
        super().__init__()
        self._budget_per_step = agent_budget_per_collector_step

    def __call__(
        self, actions: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self._calculate_funding(actions)

    @abstractmethod
    def _calculate_funding(
        self, actions: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError()

    def get_action_space(
        self, num_collectors: int
    ) -> tuple[
        float | npt.NDArray[Any],  # Low
        float | npt.NDArray[Any],  # High
        Sequence[int],  # Shape
    ]:
        return (0.0, self._budget_per_step, (num_collectors,))


class PrivateFunding(Mechanism):
    def _calculate_funding(self, actions):
        contributions = np.clip(actions, 0.0, self._budget_per_step)
        return contributions.sum(axis=0), contributions.sum(axis=1)


class QuadraticFunding(Mechanism):
    def _calculate_funding(self, actions):
        contributions = np.clip(actions, 0.0, self._budget_per_step)
        h = (contributions**0.5).sum(axis=0)
        h = h**2
        return h, contributions.sum(axis=1)


class AssuranceContract(Mechanism):
    COLLECTOR_BUDGET_LOW = 0.01

    def _calculate_funding(self, actions):
        actions = actions.copy()
        num_agents = actions.shape[0]
        num_collectors = actions.shape[1] // 2
        contributions_grad = actions[:, :num_collectors]
        contributions_bound = actions[:, num_collectors:]

        contributions = np.zeros((num_agents, num_collectors))

        for collector in range(num_collectors):
            collector_fund = np.zeros(num_agents)
            collector_grad = contributions_grad[:, collector]
            collector_budget = contributions_bound[:, collector]
            collector_mask = collector_budget > self.COLLECTOR_BUDGET_LOW

            while True:
                collector_grad_current = collector_grad[collector_mask].sum()
                if collector_grad_current < 1.0:
                    break

                collector_grad_round = (
                    collector_grad[collector_mask] / collector_grad_current
                )
                max_collection_amt = (
                    collector_budget[collector_mask] / collector_grad_round
                ).min()

                transferred_funds = np.zeros((num_agents,))
                transferred_funds[collector_mask] = (
                    collector_grad_round * max_collection_amt
                )

                collector_fund += transferred_funds
                collector_budget -= transferred_funds

                collector_mask = collector_budget > self.COLLECTOR_BUDGET_LOW

            contributions[:, collector] = collector_fund

        return contributions.sum(axis=0), contributions.sum(axis=1)

    def get_action_space(self, num_collectors):
        return (
            0.0,
            np.array(
                [1.0] * num_collectors + [self._budget_per_step] * num_collectors,
                dtype=np.float32,
            ),
            (num_collectors * 2,),
        )
