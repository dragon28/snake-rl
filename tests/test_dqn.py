from snakerl.dqn import Config, load_policy, train
from snakerl.evaluate import evaluate


def test_tiny_training_run_saves_loadable_checkpoints(tmp_path):
    hist = train(Config(total_steps=6_000, checkpoints=(0, 6_000), learn_start=500),
                 tmp_path, log=lambda *_: None)
    assert hist and (tmp_path / "step_0006000.pt").exists()
    r = evaluate(load_policy(tmp_path / "step_0006000.pt"), range(5))
    assert r["episodes"] == 5
