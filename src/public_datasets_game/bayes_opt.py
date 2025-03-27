from typing import Any
import functools

import numpy as np
import numpy.typing as npt
import gymnasium.spaces as s
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import norm

from perlin_numpy import generate_perlin_noise_2d

from public_datasets_game.mechanism import Mechanism
from public_datasets_game.pdg import (
    Consumer,
    Collector,
    PublicDatasetsGame,
    RewardAllocationType,
    DeficitResolutionMethod,
)

Dataset = npt.NDArray[np.int32]
ObsType = npt.NDArray[np.float32]


class BayesOptCollector(Collector[Dataset]):
    def __init__(
        self,
        cost_per_play: float,
        partitions: list[npt.NDArray[np.int32]],
        probabilities: list[float],
    ):
        super().__init__()

        self._rng = np.random.default_rng()
        self._cost_per_play = cost_per_play
        assert len(probabilities) == len(partitions)
        self._probabilities = probabilities
        self._partitions = partitions
        self._hash = hash(tuple(probabilities))
        self._funds = 0.0

    def step(self, funding: float) -> Dataset:
        self._funds += funding
        num_datapoints = int(self._funds / self._cost_per_play)
        self._funds -= num_datapoints * self._cost_per_play

        if num_datapoints > 0:
            sample_partitions = self._rng.choice(
                len(self._partitions), size=num_datapoints, p=self._probabilities
            )
            datapoints = np.concatenate(
                [
                    self._rng.choice(self._partitions[p], size=1, replace=True)
                    for p in sample_partitions
                ]
            )
        else:
            datapoints = np.array([], dtype=np.int32)

        return datapoints

    def reset(self, seed):
        super().reset(seed)
        if seed is not None:
            rng_seed = (seed + self._hash) % 2**32
        else:
            rng_seed = None
        self._rng = np.random.default_rng(rng_seed)
        self._funds = 0.0


class BayesOptConsumer(Consumer[ObsType, Dataset]):
    def __init__(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        partition_points: list[npt.NDArray[np.float32]],
        initial_collection: list[npt.NDArray[np.int32]],
        collectors: list[BayesOptCollector],
    ):
        super().__init__()
        self.X = X
        self.Y = Y
        rbf = RBF(length_scale=0.2, length_scale_bounds="fixed")
        self.kernel = ConstantKernel() * rbf + WhiteKernel(
            noise_level=0.1, noise_level_bounds="fixed"
        )
        self.partition_points = partition_points
        self.collectors = collectors
        self.initial_collection = initial_collection
        self.collected = np.zeros(self.X.shape[0], dtype=np.int32)
        self.collected[initial_collection] = 1

        self._previous_best = max(self.Y[self.collected == 1])

    def compute_observation(self, datasets: list[Dataset]):
        for dataset in datasets:
            self.collected[dataset] = 1
        collected_points = self.X[self.collected == 1]

        # Train GP
        collected_values = self.Y[self.collected == 1]
        gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5)
        gpr.fit(collected_points, collected_values)
        best_y = collected_values.max()

        # Compute expected improvement for each partition
        partition_improvement = []
        for partition in self.partition_points:
            partition_improvement.append(
                expected_improvement(partition, gpr, best_y).mean()
            )

        collector_improvement: list[float] = []
        for collector in self.collectors:
            probabilities = collector._probabilities
            assert len(probabilities) == len(partition_improvement)

            ei = sum([p * v for p, v in zip(probabilities, partition_improvement)])
            collector_improvement.append(ei)

        obs = collector_improvement
        obs.append((self.collected == 1).sum())
        return np.array(collector_improvement, dtype=np.float32)

    def step(self, datasets: list[Dataset]):
        obs = self.compute_observation(datasets)

        new_best = self.Y[self.collected == 1].max()
        r = new_best - self._previous_best

        self._previous_best = new_best

        return obs, r, {}

    def reset(self, seed):
        self.collected = np.zeros(self.X.shape[0], dtype=np.int32)
        self.collected[self.initial_collection] = 1
        self._previous_best = max(self.Y[self.collected == 1])

        return self.compute_observation([]), {}


class BayesOptGame(PublicDatasetsGame[ObsType, Dataset]):
    GRANULARITY = 50

    def __init__(
        self,
        num_consumers: int,
        mechanism: Mechanism,
        reward_allocation: RewardAllocationType = "individual",
        deficit_resolution: DeficitResolutionMethod = "tax",
        max_steps: int = 500,
        infinite_horizon: bool = True,
        normalise_action_space: bool = False,
        randomise_on_reset: bool = True,
        cost_per_play: float = 0.5,
        return_funds_info: bool = True,
    ):
        num_collectors = 4
        self.rng = np.random.default_rng()
        self.randomise_on_reset = randomise_on_reset

        x1 = np.linspace(0, 1 - 1 / self.GRANULARITY, self.GRANULARITY)
        x2 = np.linspace(0, 1 - 1 / self.GRANULARITY, self.GRANULARITY)
        x1, x2 = np.meshgrid(x1, x2)
        X = np.stack((x1.flatten(), x2.flatten()), axis=1)

        top_left = np.where((X[:, 0] < 0.5) & (X[:, 1] >= 0.5))[0]
        top_right = np.where((X[:, 0] >= 0.5) & (X[:, 1] >= 0.5))[0]
        bottom_left = np.where((X[:, 0] < 0.5) & (X[:, 1] < 0.5))[0]
        bottom_right = np.where((X[:, 0] >= 0.5) & (X[:, 1] < 0.5))[0]
        partitions = [top_left, top_right, bottom_left, bottom_right]
        partition_points = [X[p] for p in partitions]

        corners = [
            np.where((X[:, 0] == 0) & (X[:, 1] == 0))[0][0],
            np.where((X[:, 0] == 0.98) & (X[:, 1] == 0))[0][0],
            np.where((X[:, 0] == 0) & (X[:, 1] == 0.98))[0][0],
            np.where((X[:, 0] == 0.98) & (X[:, 1] == 0.98))[0][0],
        ]

        self.consumers: list[BayesOptConsumer] = []
        self.collectors: list[BayesOptCollector] = []

        for i in range(num_collectors):
            probabilities = [0.1, 0.1, 0.1, 0.1]
            probabilities[i] = 0.7

            self.collectors.append(
                BayesOptCollector(
                    cost_per_play=cost_per_play,
                    probabilities=probabilities,
                    partitions=partitions,
                )
            )

        for _ in range(num_consumers):
            Y = self._generate_random_objective()
            self.consumers.append(
                BayesOptConsumer(
                    X=X,
                    Y=Y,
                    partition_points=partition_points,
                    initial_collection=corners,
                    collectors=self.collectors,
                )
            )

        super().__init__(
            self.consumers,
            self.collectors,
            mechanism=mechanism,
            reward_allocation=reward_allocation,
            deficit_resolution=deficit_resolution,
            max_steps=max_steps,
            infinite_horizon=infinite_horizon,
            normalise_action_space=normalise_action_space,
            return_funds_info=return_funds_info,
        )

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)

        if self.randomise_on_reset or seed is not None:
            for i in range(self.num_agents):
                Y = self._generate_random_objective()
                self.consumers[i].Y = Y

        obs, info = super().reset(seed, options)
        return obs, info

    def _generate_random_objective(self) -> npt.NDArray[np.float32]:
        np.random.seed(self.rng.integers(0, 2**32))
        Y = (
            generate_perlin_noise_2d((self.GRANULARITY, self.GRANULARITY), (2, 2))
            .astype(np.float32)
            .flatten()
            * 20
        )

        Y += self.rng.normal(0, 0.1, size=Y.shape)

        return Y  # type: ignore

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return s.Box(-np.inf, np.inf, shape=(self.num_collectors + 1,))


def expected_improvement(
    X: npt.NDArray[Any], gpr: GaussianProcessRegressor, best_y: float
):
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = np.clip(sigma, 1e-9, None)
    improvement = mu - best_y
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei.reshape(-1, 1)
