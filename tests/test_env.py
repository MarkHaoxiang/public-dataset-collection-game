import pytest
from pettingzoo.test import parallel_api_test


@pytest.mark.parametrize("num_bandits", [2, 3])
@pytest.mark.parametrize("num_arms", [2, 3])
def test_rotting_bandits(num_bandits: int, num_arms: int):
    from public_datasets_game.rotting_bandits import RottingBanditsGame
    from public_datasets_game.mechanism import PrivateFunding

    env = RottingBanditsGame(
        num_arms=num_arms, num_bandits=num_bandits, mechanism=PrivateFunding()
    )
    parallel_api_test(env, num_cycles=100)
