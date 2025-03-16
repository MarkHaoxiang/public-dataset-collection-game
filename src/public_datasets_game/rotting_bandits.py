from typing import Any
import functools

import numpy as np
import numpy.typing as npt
import gymnasium.spaces as s

from public_datasets_game.pdg import (
    Collector,
    Consumer,
    Mechanism,
    PublicDatasetsGame,
    Reward,
    Info,
    AgentID,
)


Dataset = int
ObsType = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None


class RottingBanditsConsumer(Consumer[ObsType, Dataset]):
    def __init__(self, k: list[int], theta: list[float], sigma: list[float]):
        """Models are of the form

        (n/k + 1)^(-theta) + N(0, theta)
        """
        super().__init__()

        self.k = k
        self.theta = theta
        self.sigma = sigma

        self.num_arms = len(k)
        assert self.num_arms == len(theta)
        assert self.num_arms == len(sigma)

    def reset(
        self,
        seed: int | None,
    ) -> tuple[ObsType, dict[Any, Any]]:
        self.rng = np.random.default_rng(seed)
        self.num_plays_counter = [0 for _ in range(self.num_arms)]

        return None, {}

    def step(self, datasets: list[Dataset]) -> tuple[ObsType, Reward, Info]:
        obs = self.compute_observation(datasets)
        if obs is None:
            return None, 0.0, {}

        for arm, num_plays in enumerate(datasets):
            self.num_plays_counter[arm] += num_plays
        utility = obs[1].sum()

        return obs, utility, {}

    def compute_observation(self, datasets: list[Dataset]) -> ObsType:
        plays = []
        returns = []

        for arm, num_plays in enumerate(datasets):
            k = self.k[arm]
            theta = self.theta[arm]
            sigma = self.sigma[arm]

            n_start = self.num_plays_counter[arm]
            n = np.linspace(n_start + 1, n_start + num_plays, num=num_plays)

            plays.append(np.full(num_plays, arm))
            returns.append(
                (n / k + 1) ** -theta + self.rng.normal(0, sigma, size=num_plays)
            )

        plays_concat = np.concatenate(plays)
        returns_concat = np.concatenate(returns)
        obs = (plays_concat, returns_concat)

        return obs


class RottingBanditsCollector(Collector[Dataset]):
    def __init__(self, cost_per_play: float):
        super().__init__()
        self.cost_per_play = cost_per_play

    def step(self, funding: float) -> Dataset:
        self._funds += funding
        num_datapoints = int(self._funds / self.cost_per_play)
        self._funds -= num_datapoints * self.cost_per_play

        return num_datapoints

    def reset(self, seed):
        super().reset(seed)
        self._funds = 0.0


class RottingBanditsGame(PublicDatasetsGame[ObsType, Dataset]):
    def __init__(
        self,
        num_bandits: int,
        num_arms: int,
        mechanism: Mechanism,
        max_steps: int = 500,
        cost_per_play: float = 0.1,
        infinite_horizon: bool = True,
    ):
        self.rng = np.random.default_rng()
        self.consumers: list[RottingBanditsConsumer] = []
        self.collectors: list[RottingBanditsCollector] = []
        self.num_arms = num_arms
        for _ in range(num_bandits):
            k, theta, sigma = self._generate_random_agent(num_arms)
            consumer = RottingBanditsConsumer(k, theta, sigma)
            self.consumers.append(consumer)

        for _ in range(num_arms):
            collector = RottingBanditsCollector(cost_per_play)
            self.collectors.append(collector)

        super().__init__(
            consumers=self.consumers,
            collectors=self.collectors,
            mechanism=mechanism,
            dataset_update_method="replace",
            max_steps=max_steps,
            infinite_horizon=infinite_horizon,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        # Reset the distribution of arms for each consumer
        for i in range(self.num_agents):
            k, theta, sigma = self._generate_random_agent(self.num_arms)
            self.consumers[i].k = k
            self.consumers[i].theta = theta
            self.consumers[i].sigma = sigma

        return obs, info

    def _generate_random_agent(self, num_arms: int):
        k = [100 for _ in range(num_arms)]
        theta: list[float] = self.rng.uniform(0.1, 1.0, size=num_arms).tolist()
        sigma: list[float] = self.rng.uniform(0.1, 0.5, size=num_arms).tolist()
        return k, theta, sigma

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        return s.Tuple(
            (s.Sequence(s.Box(-np.inf, np.inf)), s.Sequence(s.Box(-np.inf, np.inf)))
        )
