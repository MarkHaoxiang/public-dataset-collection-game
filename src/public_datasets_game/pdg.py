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


class Consumer[ObsType, Dataset]:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self, datasets: list[Dataset]) -> tuple[ObsType, Reward, Info]:
        raise NotImplementedError()

    @abstractmethod
    def compute_observation(self, datasets: list[Dataset]) -> ObsType:
        raise NotImplementedError()

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
        dataset_update_method: Literal["extend", "replace"] = "replace",
        max_steps: int = 500,
        infinite_horizon: bool = True,
        agent_budget_per_collector_step: float = 100.0,
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

        self._dataset_update_method = dataset_update_method
        self._max_steps = max_steps
        self._infinite_horizon = infinite_horizon
        self._agent_budget_per_collector_step = agent_budget_per_collector_step

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

        # Agent contributions
        contributions = np.zeros(
            (self.num_agents, self.num_collectors), dtype=np.float32
        )
        for i, agent in enumerate(actions.keys()):
            action = actions[agent]
            assert action.shape == (self.num_collectors,), (
                f"Invalid action shape {action.shape} but expected {(self.num_collectors,)}"
            )
            contributions[i] = np.clip(
                action, 0.0, self._agent_budget_per_collector_step
            )

        # Agent funding
        funding = self.mechanism(contributions)
        assert funding.shape == (self.num_collectors,), (
            f"Invalid funding shape {funding.shape} but expected {(self.num_collectors,)}"
        )

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
        rewards = {k: v[1] - actions[k].sum() for k, v in training_results.items()}
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
        return s.Box(
            low=0.0,
            high=self._agent_budget_per_collector_step,
            shape=(self.num_collectors,),
            dtype=np.float32,
        )

    @property
    def num_collectors(self) -> int:
        return len(self.collectors)
