from __future__ import annotations

import numpy as np

from snake_rl.agent import SnakeAgent


def test_obs_to_state_allows_front_cell_when_it_is_tail_candidate():
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[2, 2] = 2  # head

    # Body chain from neck to tail:
    # neck (2,1) -> (1,1) -> (1,2) -> (1,3) -> tail (2,3)
    for y, x in ((2, 1), (1, 1), (1, 2), (1, 3), (2, 3)):
        grid[y, x] = 1

    grid[4, 4] = 3  # apple
    obs = {"grid": grid, "dir": np.array([1, 0], dtype=np.int8)}  # moving right

    state = SnakeAgent.obs_to_state(obs)
    danger_front = state[0]
    assert danger_front == 0
