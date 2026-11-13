.PHONY: setup test lint train evaluate gif demo all
setup:
	uv sync
lint:
	uv run ruff check .
test:
	uv run pytest -q
train:
	for s in 0 1 2; do uv run snakerl train --seed $$s --out runs/seed$$s; done
evaluate:
	uv run snakerl evaluate --runs runs/seed0 runs/seed1 runs/seed2
gif:
	uv run snakerl gif --run runs/seed0 --out docs/training.gif
demo:  # play the committed final model in a GIF, no training needed
	uv run snakerl gif --run models --out demo.gif
all: lint test train evaluate gif
