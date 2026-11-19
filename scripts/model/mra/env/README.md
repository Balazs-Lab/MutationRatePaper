# `pymc_env` — conda environment for the mra model

This directory captures the exact conda environment (`pymc_env`) used to fit the
paper's primary mutation-supply model (`mra`) and run all downstream analyses.
It is provided so the fit is reproducible on another machine.

**Environment used for the paper**

- Python **3.13.9**, PyMC **5.26.1**, ArviZ **0.22.0**, PyTensor **2.35.1**,
  NumPy **2.3.4**, pandas **2.3.3**, patsy **1.0.2**
- Built with conda 26.3.2 on macOS (arm64 / Apple Silicon), channel `conda-forge`.

## Files

| File | What it is | Use when |
|------|------------|----------|
| `pymc_env.from-history.yml` | Only the packages explicitly requested (no transitive deps, no versions pinned beyond `pymc>=5`). Most portable across OS/architecture. | **Recommended** for recreating on a different OS/CPU (e.g. Linux, Intel Mac). |
| `pymc_env.no-builds.yml` | Full solved dependency list with versions but no build strings. | Reproduce the same versions on a compatible platform without over-pinning build hashes. |
| `pymc_env.full.yml` | Complete solved environment including exact build strings. | Byte-for-byte reproduction on the **same** platform (macOS arm64). |
| `pymc_env.pip-freeze.txt` | `pip freeze` of the environment (for reference / pip-only setups). | Cross-check exact versions. |

The machine-specific `prefix:` line has been removed from the YAML files.

## Recreate the environment

Recommended (portable):

```bash
conda env create -n pymc_env -f pymc_env.from-history.yml
conda activate pymc_env
```

To match the paper's exact package versions on a compatible platform:

```bash
conda env create -n pymc_env -f pymc_env.no-builds.yml
conda activate pymc_env
```

Exact rebuild on macOS arm64 (same platform as the original):

```bash
conda env create -n pymc_env -f pymc_env.full.yml
conda activate pymc_env
```

> Mamba works as a faster drop-in replacement for `conda` in any of the commands
> above.

## Verify

```bash
conda run -n pymc_env python -c "import pymc, arviz; print(pymc.__version__, arviz.__version__)"
# expect: 5.26.1 0.22.0
```

Then follow `../mra_RUNBOOK.md` to build the input, fit the model, and reproduce
the outputs.
