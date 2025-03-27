from abc import ABC, abstractmethod
import functools
from collections.abc import Sequence
from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from pettingzoo import ParallelEnv
import gymnasium.spaces as s

from public_datasets_game.mechanism import Mechanism

AgentID = str
Reward = float
Terminated = bool
Truncated = bool
ActionType = npt.NDArray[np.float32]
Info = dict[AgentID, dict[Any, Any]]

RewardAllocationType = Literal["individual", "collaborative"]
DeficitResolutionMethod = Literal["tax", "ignore"]
DatasetUpdateMethod = Literal["extend", "replace"]


class Consumer[ObsType, Dataset]:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self, datasets: list[Dataset]) -> tuple[ObsType, Reward, Info]:
        raise NotImplementedError()

    @abstractmethod
    def compute_observation(self, datasets: list[Dataset]) -> ObsType:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, seed: int | None) -> tuple[ObsType, dict[Any, Any]]:
        raise NotImplementedError()


class Collector[Dataset]:
    @abstractmethod
    def step(self, funding: float) -> Dataset:
        raise NotImplementedError()

    def reset(self, seed: int | None) -> None:
        pass


class PublicDatasetsGame[ObsType, Dataset](
    ParallelEnv[AgentID, ObsType, ActionType], ABC
):
    def __init__(
        self,
        consumers: Sequence[Consumer[ObsType, Dataset]],
        collectors: Sequence[Collector[Dataset]],
        mechanism: Mechanism,
        reward_allocation: RewardAllocationType = "individual",
        dataset_update_method: DatasetUpdateMethod = "replace",
        deficit_resolution: DeficitResolutionMethod = "tax",
        max_steps: int = 500,
        infinite_horizon: bool = True,
        normalise_action_space: bool = False,
    ):
        super().__init__()

        self.consumers = consumers
        self.collectors = collectors
        self.mechanism = mechanism

        self.agents = [f"agent_{i}" for i in range(len(consumers))]
        self.possible_agents = self.agents
        self.agent_to_consumer = {
            agent: self.consumers[i] for i, agent in enumerate(self.agents)
        }

        self._deficit_resolution = deficit_resolution
        self._reward_allocation = reward_allocation
        self._dataset_update_method = dataset_update_method
        self._max_steps = max_steps
        self._infinite_horizon = infinite_horizon

        self._normalise_action_space = normalise_action_space
        self.action_space_settings = self.mechanism.get_action_space(
            self.num_collectors
        )

        # Stateful
        self._step = 0
        self._datasets: list[Dataset] = []

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[dict[AgentID, ObsType], Info]:
        obs: dict[AgentID, ObsType] = {}
        info: Info = {}
        for agent, consumer in zip(self.agents, self.consumers):
            agent_obs, agent_info = consumer.reset(seed)
            obs[agent] = agent_obs
            info[agent] = agent_info
        for collector in self.collectors:
            collector.reset(seed)

        self._step = 0

        return obs, info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, Reward],
        dict[AgentID, Terminated],
        dict[AgentID, Truncated],
        Info,
    ]:
        if self._normalise_action_space:
            for agent in actions.keys():
                action = actions[agent]
                action -= self.action_space_settings[0]
                actions[agent] = action * (
                    self.action_space_settings[1] - self.action_space_settings[0]
                )

        cont = {agent: False for agent in self.agents}
        finish = {agent: True for agent in self.agents}
        if self._step >= self._max_steps:
            if self._dataset_update_method == "replace":
                self._datasets = []
            return (
                self._observe_all(),
                {agent: 0.0 for agent in self.agents},
                cont if self._infinite_horizon else finish,
                finish,
                {agent: {} for agent in self.agents},
            )
        self._step += 1

        # Funding public goods
        action_shape = self.mechanism.get_action_space(self.num_collectors)[2]
        joint_actions = np.zeros(
            (self.num_agents, *action_shape),
            dtype=np.float32,
        )
        agent_idx_map: dict[AgentID, int] = {}
        for i, (agent, action) in enumerate(actions.items()):
            assert action.shape == action_shape, (
                f"Invalid action shape {action.shape} but expected {action_shape}"
            )
            joint_actions[i] = action
            agent_idx_map[agent] = i

        funding, contributions = self.mechanism(joint_actions)
        assert contributions.shape == (self.num_agents,)
        assert funding.shape == (self.num_collectors,)

        contributions_total = contributions.sum()
        funding_total = funding.sum()
        tax_per_agent = 0
        deficit = funding_total - contributions_total
        if deficit > 0:
            match self._deficit_resolution:
                case "tax":
                    tax_per_agent = deficit / self.num_agents
                case "ignore":
                    tax_per_agent = 0

        # Step collectors
        datasets = [
            collector.step(funding[i]) for i, collector in enumerate(self.collectors)
        ]

        match self._dataset_update_method:
            case "extend":
                self._datasets.extend(datasets)
            case "replace":
                self._datasets = datasets

        # Training
        training_results = {
            agent: self._get_consumer(agent).step(self._datasets)
            for agent in self.agents
        }

        obs = {k: v[0] for k, v in training_results.items()}

        match self._reward_allocation:
            case "individual":
                rewards = {
                    k: v[1] - contributions[agent_idx_map[k]] - tax_per_agent
                    for k, v in training_results.items()
                }
            case "collaborative":
                team_reward = sum(
                    [
                        v[1] - contributions[agent_idx_map[k]]
                        for k, v in training_results.items()
                    ]
                )
                team_reward /= self.num_agents
                team_reward -= tax_per_agent
                rewards = {k: team_reward for k in training_results.keys()}

        infos = {k: v[2] for k, v in training_results.items()}

        end = self._step >= self._max_steps
        truncated = finish if end else cont
        terminated = finish if end and not self._infinite_horizon else cont

        return obs, rewards, terminated, truncated, infos

    def observe(self, agent: AgentID) -> ObsType:
        consumer = self._get_consumer(agent)
        return consumer.compute_observation(self._datasets)

    def _observe_all(self) -> dict[AgentID, ObsType]:
        return {agent: self.observe(agent) for agent in self.agents}

    def _get_consumer(self, agent: AgentID) -> Consumer[ObsType, Dataset]:
        return self.agent_to_consumer[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        low, high, shape = self.action_space_settings

        if self._normalise_action_space:
            low = 0.0
            high = 1.0

        return s.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32,
        )

    @property
    def num_collectors(self) -> int:
        return len(self.collectors)
