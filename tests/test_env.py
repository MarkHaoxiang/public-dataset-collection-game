import pytest
from pettingzoo.test import parallel_api_test
from public_datasets_game.mechanism import (
    Mechanism,
    PrivateFunding,
    QuadraticFunding,
    AssuranceContract,
)


@pytest.mark.parametrize("num_bandits", [2, 3])
@pytest.mark.parametrize("num_arms", [2, 3])
@pytest.mark.parametrize(
    "mechanism", [PrivateFunding(), QuadraticFunding(), AssuranceContract()]
)
def test_rotting_bandits(num_bandits: int, num_arms: int, mechanism: Mechanism):
    from public_datasets_game.rotting_bandits import RottingBanditsGame

    env = RottingBanditsGame(
        num_arms=num_arms, num_bandits=num_bandits, mechanism=mechanism
    )
    parallel_api_test(env, num_cycles=100)
