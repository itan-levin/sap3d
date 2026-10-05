#!/usr/bin/env bash
# Builds the CUDA extensions against the torch installed in the active environment.
# Needs a CUDA toolkit whose major version matches torch's (CUDA_HOME pointing at it).
# FORCE_CUDA=1 makes diso build its CUDA kernels even where no GPU is visible; without a GPU and
# without it, diso silently builds a CPU-only module that the DMC surface extractor cannot use.
set -euo pipefail

FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps --no-binary diso diso==0.1.4
python -c "import diso; from diso import DiffDMC; print('diso', diso.__file__)"
