# SAP3D

Localized editing of 3D assets with per-object personalized adapters on top of Tencent's
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape model. Given a source mesh, its
adapter, an edited view of the source and a 3D region, the edit stage regenerates the region to match
the edited view and keeps the rest of the object.

The repository is organized as independent stages: personalize (asset to adapter), edit (this release),
texture and eval.

## Install

Python 3.10 with a CUDA GPU. Two pinned environments are provided; each brings the CUDA compiler that
`build_extensions.sh` needs:

```bash
conda env create -f environment.yml            # PyTorch 2.5.1, CUDA 12.4
conda activate sap3d
bash scripts/build_extensions.sh               # builds diso against the installed torch
pip install -e .
```

or `environment-torch28.yml` for PyTorch 2.8.0 (CUDA 12.8), which also pins `setuptools<81`.

The shape model weights (`tencent/Hunyuan3D-2.1`) are downloaded from Hugging Face on first use into
`~/.cache/hy3dgen` (override with `HY3DGEN_MODELS`); the background remover downloads its model on
first use.

## Edit

A benchmark case from [Edit3D-Bench](https://huggingface.co/datasets/huanngzh/Edit3D-Bench):

```bash
sap3d-edit --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2 \
           --adapters adapters/ --out outputs/sorting_bus_p2
```

Any asset with its own adapter:

```bash
sap3d-edit --source model.glb --edited-view edited.png --region region.yaml \
           --orbit 15 -90 --adapter my_adapter/ --out outputs/my_edit
```

- The region is a GLB holding one closed mesh object per part, or a YAML/JSON list of boxes, spheres
  and cylinders in the source mesh's coordinates (`sap3d.region.load_region` documents the format).
  Parts should be convex and must not overlap; the 3D probe samples each part's convex hull.
- The edited view is a PNG, square when the mask is projected. Without real transparency its
  background must be near-white.
- The 2D mask is either given (`--mask`, single-channel, the edited view's size) or projected from the
  region through the edited view's camera. Projection keeps the pixels where the region is not hidden
  behind the source mesh. The camera is one of:
  - `--camera-meta meta.json --view K`: entry K of a file in the benchmark's `meta.json` format;
  - `--camera camera.json`: `{"transform_matrix": <4x4 Blender camera-to-world, Z up>,
    "camera_angle_x": <horizontal field of view, radians>, "size": <pixels>}`;
  - `--orbit ELEVATION AZIMUTH`: the benchmark renderer's camera sphere (radius 1.8, 49.13° field of view,
    azimuth -90° for its first view), which frames an asset normalized like the benchmark's: centered at
    the origin with its longest side 1.
- `--global` uses the source's bounding box, padded by 0.1 of its longest side, as the region, and its
  projection as the mask. The global cases behind the paper's figures were built differently, with
  their own region and mask files.
- `--config configs/edit_global_uniform.yaml` runs a global edit with one adapter strength for every
  token instead of the local schedule.

The output is `edit.glb` in the source mesh's frame, `edit.json` (settings, orientation searches, region
size per step), `regions.npz` (the latent tokens each probe nominated at each step) and, when the mask
was projected, `mask_2d.png`.

`configs/edit.yaml` lists every setting with its default value; `sap3d-edit --dump-config` prints the
resolved settings.

### What the edit stage does beyond the paper's description

- An edited view without real transparency gets its background removed, with every pixel that is not
  near-white raised to opaque.
- A preliminary uniform-strength generation, sharing the edit's first steps, is searched for a quarter
  turn about the vertical relative to the source outside the region; the region is turned by the inverse
  before the probes read it. The finished mesh gets the same search. Both are skipped when the region
  contains the whole source.

## Tests

```bash
python tests/run_all.py
```

CPU only; run from the repository root.

## License

The Tencent Hunyuan 3D 2.1 Community License Agreement (`LICENSE`, `NOTICE`). `third_party/hy3dshape` is
a modified copy of Tencent's package; its changes are listed in `third_party/hy3dshape/MODIFICATIONS.md`.
