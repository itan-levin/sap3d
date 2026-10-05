#!/usr/bin/env bash
# Builds the compiled extensions against the torch installed in the active environment, from the
# repository root:
#   diso               the differentiable marching cubes the shape decoder extracts surfaces with (CUDA)
#   custom_rasterizer  the paint model's rasterizer (CUDA)
#   mesh painter       the paint model's texture inpainting helper (C++), compiled in place into
#                      third_party/hy3dpaint/DifferentiableRenderer, where hy3dpaint imports it from
# Needs a CUDA toolkit whose major version matches torch's (CUDA_HOME pointing at it) and a C++ compiler.
# FORCE_CUDA=1 makes diso build its CUDA kernels even where no GPU is visible; without a GPU and
# without it, diso silently builds a CPU-only module that the DMC surface extractor cannot use.
# --no-build-isolation keeps pip from building against a second, freshly downloaded torch.
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps --no-binary diso diso==0.1.4
python -c "import diso; from diso import DiffDMC; print('diso', diso.__file__)"

python -m pip install --no-build-isolation --no-deps ./third_party/hy3dpaint/custom_rasterizer
python -c "import torch, custom_rasterizer; print('custom_rasterizer', custom_rasterizer.__file__)"

(
  cd third_party/hy3dpaint/DifferentiableRenderer
  suffix=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
  c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp \
      -o "mesh_inpaint_processor${suffix}"
)
python -c "import sys; sys.path.insert(0, 'third_party/hy3dpaint'); \
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint; print('mesh painter built')"
