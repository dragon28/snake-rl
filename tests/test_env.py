import numpy as np
from gymnasium.utils.env_checker import check_env

from snakerl.env import SnakeEnv
from snakerl.evaluate import evaluate
from snakerl.policies import greedy_policy, random_policy


def test_gymnasium_api():
    check_env(SnakeEnv(), skip_render_check=True)


def test_wall_collision_ends_episode():
    env = SnakeEnv(10)
    env.reset(seed=0)
    env.food = (0, 0)
    done = False
    for _ in range(10):
        _, r, done, _, _ = env.step(0)
        if done:
            break
    assert done and r == -10.0


def test_eating_grows_and_food_never_on_body():
    env = SnakeEnv(10)
    env.reset(seed=0)
    hx, hy = env.snake[0]
    env.food = (hx + 1, hy)
    _, r, _, _, info = env.step(0)
    assert r == 10.0 and info["length"] == 4
    assert env.food not in env.body


def test_moving_into_departing_tail_is_legal():
    env = SnakeEnv(10)
    env.reset(seed=0)
    from collections import deque
    env.snake = deque([(5, 5), (5, 6), (4, 6), (4, 5)])  # head, ..., tail at (4,5)
    env.body = set(env.snake)
    env.heading = 2  # moving down; turn right -> left (x-1), into tail cell
    env.food = (0, 0)
    _, _, term, _, _ = env.step(1)
    assert not term


def test_seed_determinism():
    def roll():
        env = SnakeEnv()
        env.reset(seed=7)
        rng = np.random.default_rng(1)
        return [env.step(int(rng.integers(3)))[1] for _ in range(50) if not env.dead]
    assert roll() == roll()


def test_greedy_beats_random():
    seeds = range(20)
    assert evaluate(greedy_policy, seeds)["mean"] > evaluate(
        random_policy(np.random.default_rng(0)), seeds)["mean"]
