# Sandeha

Sandeha (Sanskrit for "doubt") is a neural 2D smoke simulator that measures its own doubt and hands control to a conventional physics solver when that doubt is too high.

**Status:** project setup. No models have been trained and no results exist yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a driver that supports CUDA 13.2.

```bash
uv sync
uv run pytest
```

## Long jobs

```bash
scripts/long_job.sh <name> <command...>
```

The wrapper refuses to start without mains power, holds a sleep and lid inhibitor while the command runs, and logs to `runs/<name>.log` and `runs/jobs.log`.

## Documents

See `docs/`. Start with `docs/PRD.md`, `docs/HLD.md` and `docs/DESIGN.md`.
