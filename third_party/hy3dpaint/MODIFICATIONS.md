# Changes to Tencent's hy3dpaint package

`hy3dpaint/` holds the parts of Tencent's Hunyuan3D-2.1 `hy3dpaint/` directory that texture painting
needs. No carried file is changed.

## Carried unchanged

`textureGenPipeline.py`, `hunyuanpaintpbr/`, `DifferentiableRenderer/` (sources; the mesh painter is
compiled in place by `scripts/build_extensions.sh`), `custom_rasterizer/`, `utils/`, `cfgs/`, `LICENSE`.

## Added

| file | content |
|---|---|
| `custom_rasterizer/pyproject.toml` | Declares the build requirements (setuptools, wheel, torch) so the extension can be built with `pip install --no-build-isolation`. |

## Not carried

`demo.py`, `train.py`, `convert_utils.py`, `src/` (training data loading),
`train_examples/`, `assets/` and `README.md`.
