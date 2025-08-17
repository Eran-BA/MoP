# Contributing to MoP
- Create a virtualenv, `pip install -r requirements.txt && pip install -e .`
- Run tests: `pytest -q`
- Smoke run (CPU/MPS ok): `python experiments/cifar10_multi_seed.py --tiny --steps 400 --eval_every 100 --seeds 0`
- Lint/format: coming soon (pre-commit)
- Open PRs against `main` with a clear description and before/after numbers if relevant.
