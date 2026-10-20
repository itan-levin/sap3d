# SAP3D

[Project page](https://itan-levin.github.io/sap3d-page/)

SAP3D edits an existing 3D asset from an edited 2D view of it, on top of Tencent's
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape model. A low-rank adapter is first
fitted to the asset, so that the generator reproduces it; during editing the adapter is applied at a different
strength per latent token, off for the tokens of the edit region and on for all others, so the region follows
the edited view while the rest of the asset is kept. Two probes, one on the transformer's image
cross-attention and one on the shape decoder's cross-attention, decide which tokens belong to the region at
every denoising step.

The code is organized as independent stages:

| stage | command | input | output |
|---|---|---|---|
| personalize | `sap3d-personalize` | an asset | its adapter |
| edit | `sap3d-edit` | asset, adapter, edited view, 3D region | edited mesh |
| texture (optional) | `sap3d-texture` | edited mesh, asset, edited view | textured mesh |
| eval (optional) | `sap3d-eval` | textured mesh | preservation, edit fidelity, Edit3D-Bench metrics |

`sap3d-bench` runs edit, texture and eval over a list of Edit3D-Bench cases. Adapters for the 100 Edit3D-Bench
objects can be downloaded, so the personalize stage is needed only for new assets.

## Install

Linux, an NVIDIA GPU, conda and a system C++ compiler (gcc and g++). The conda environment (`environment.yml`)
pins Python 3.10, PyTorch 2.8.0 and the CUDA 12.8 compiler.

```bash
git clone https://github.com/itan-levin/sap3d.git && cd sap3d

conda env create -f environment.yml
conda activate sap3d

export CUDA_HOME=$CONDA_PREFIX CPATH=$CONDA_PREFIX/targets/x86_64-linux/include \
       LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib
bash scripts/build_extensions.sh     # diso, custom_rasterizer, the mesh painter
pip install -e .
python scripts/download_realesrgan.py
```

- `scripts/build_extensions.sh` compiles the CUDA and C++ extensions against the installed torch with the
  environment's CUDA compiler and the system C++ compiler, for the visible GPU. Set `TORCH_CUDA_ARCH_LIST`
  (for example `"8.0;8.6;9.0;12.0"`) to build for other GPUs.
- Use an editable install from a clone (or run from the repository root with
  `PYTHONPATH=$PWD:$PWD/third_party/hy3dshape`): the texture and eval stages use the vendored paint model and
  Edit3D-Bench scorer under `third_party/` by path.
- `scripts/download_realesrgan.py` fetches `RealESRGAN_x4plus.pth`, which the paint model needs, from the
  Real-ESRGAN v0.1.0 release (https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
  to `third_party/hy3dpaint/ckpt/`.
- Model weights download on first use. From Hugging Face: `tencent/Hunyuan3D-2.1` (shape and paint models),
  `facebook/dinov2-large`, `facebook/dinov2-giant`, `facebook/dinov2-base` and `openai/clip-vit-base-patch32`.
  The LPIPS AlexNet comes from download.pytorch.org and the background remover's `u2net.onnx` from GitHub.
  Behind a proxy that blocks Hugging Face's Xet storage, set `HF_HUB_DISABLE_XET=1`.
- GPU memory, measured on an 80 GB A100: training an adapter peaks at about 62 GiB, texturing at about 21 GiB.

## Data

```bash
huggingface-cli download huanngzh/Edit3D-Bench --repo-type dataset --local-dir Edit3D-Bench
python scripts/download_adapters.py --out adapters --all          # or --objects GSO/SORTING_BUS ...
python scripts/download_global_cases.py --edit3d-bench Edit3D-Bench/data --out global_edits
```

- `download_adapters.py` downloads the adapters of the 100 Edit3D-Bench objects and writes
  `adapters/manifest.json`, which `--adapters adapters` reads. `--list` prints what is available.
- `download_global_cases.py` assembles the 12 whole-object edit cases as a benchmark folder (edited views,
  masks, regions and cameras, with each object's source model linked from Edit3D-Bench) and writes their ids to
  `global_edits/cases.txt`.

## Personalize

```bash
sap3d-personalize prepare --source asset.glb --out assets/my_asset     # not needed for an Edit3D-Bench object
sap3d-personalize dataset --asset assets/my_asset --out data/my_asset
sap3d-personalize train   --data data/my_asset --out runs/my_asset     # the adapter is runs/my_asset/adapter
```

- `prepare` puts a glTF/GLB asset with static meshes into Edit3D-Bench's layout: `source_model/model.glb`,
  centered at the origin with its longest side 1, and 16 views with their cameras (`render/render_0000-0015.png`,
  `render/meta.json`). An Edit3D-Bench object folder (`Edit3D-Bench/data/GSO/SORTING_BUS`) already has this
  layout.
- `dataset` renders the training views and samples the asset's surface. `--seed` makes the run repeatable.
- `train` fits the adapter, saves it every 500 steps under `steps/`, scores each saved step on a generated
  mesh, and copies the best one to `adapter/`. `training.json` records the losses and scores.
  `sap3d-personalize select --run runs/my_asset` chooses again, for instance after an interrupted run.
- `configs/personalize.yaml` lists every setting; `dataset` and `train` take `--config` to override it and
  `--dump-config` to print the resolved settings.

## Edit

A benchmark case:

```bash
sap3d-edit --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2 \
           --adapters adapters --out outputs/sorting_bus_p2
```

Any asset with its own adapter:

```bash
sap3d-edit --source model.glb --edited-view edited.png --region region.yaml \
           --orbit 15 -90 --adapter runs/my_asset/adapter --out outputs/my_edit
```

- The region is a GLB holding one closed mesh object per part, or a YAML/JSON list of boxes, spheres
  and cylinders in the source mesh's coordinates (`sap3d.region.load_region` documents the format).
  Parts should be convex and must not overlap.
- The edited view is a PNG, square when the mask is projected. Without transparency its background must be
  near-white.
- The 2D mask is either given (`--mask`, single-channel, the edited view's size) or projected from the
  region through the edited view's camera, which is one of:
  - `--camera-meta meta.json --view K`: entry K of a file in the benchmark's `meta.json` format;
  - `--camera camera.json`: `{"transform_matrix": <4x4 Blender camera-to-world, Z up>,
    "camera_angle_x": <horizontal field of view, radians>, "size": <pixels>}`;
  - `--orbit ELEVATION AZIMUTH`: the camera sphere of the benchmark's renderer (radius 1.8, 49.13° field of
    view, azimuth -90° for its first view), which frames an asset prepared as above.
- Whole-object (global) edits have two modes:
  - the default configuration with a region covering the whole object, either the case's own region and mask
    files or `--global`, which uses the source's bounding box padded by 0.1 of its longest side and its
    projection;
  - `--config configs/edit_global_uniform.yaml`: one adapter strength (0.5) for every token, with no region.

The output is `edit.glb` in the source mesh's frame, `edit.json` (settings and per-step region sizes),
`regions.npz` (the region's latent tokens at each step) and, when the mask was projected, `mask_2d.png`.
`configs/edit.yaml` lists every setting; `sap3d-edit --dump-config` prints the resolved settings.

## Texture

```bash
sap3d-texture --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2 \
              --mesh outputs/sorting_bus_p2/edit.glb --out outputs/sorting_bus_p2/texture
```

or, for any asset, `--mesh edit.glb --source model.glb --edited-view edited.png --region region.yaml` plus the
edited view's camera (`--camera-meta meta.json --view K`, `--camera camera.json` or `--orbit ELEVATION AZIMUTH`)
and optionally `--mask`.

- `--mode composite` (default, local edits) keeps the source's texture outside the region and paints the
  region from the edited view, then corrects the painted colors to match the edited view. It writes
  `composite.glb` and `composite_color_fit.glb`. The source must be one mesh with one base-color texture.
- `--mode generated` (global edits) paints the whole mesh and writes `generated.glb`.

`configs/texture.yaml` lists the settings.

## Eval

```bash
sap3d-eval --texture-dir outputs/sorting_bus_p2/texture --out outputs/sorting_bus_p2/eval \
           --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2
```

- **Preservation**: point-to-surface distance to the source mesh, outside the region for local edits and over
  the whole surface for global ones.
- **Edit fidelity**: DINOv2 and CLIP distances between a render through the edited view's camera and the edited
  view, inside the 2D mask.
- **Benchmark metrics** (local benchmark cases): the Edit3D-Bench scorer's Chamfer distance, PSNR, SSIM,
  LPIPS, DINO-I and CLIP-T.

## Benchmark runs

```bash
sap3d-bench --bench-root Edit3D-Bench/data --adapters adapters --out runs/bench --cases cases.txt
```

`cases.txt` holds one case id per line. The runner edits, textures and evaluates every case, writing
`runs/bench/cases/<case id>/` and `summary.json`, `summary.csv` and `summary.md` over every finished case (use
one output folder per configuration). A rerun skips finished stages whose settings and inputs match, and
refuses stages made with other settings; `--overwrite` redoes them. `--global` runs whole-object edits.

## Reproducing the paper's numbers

Use an A100-class GPU, the downloaded adapters and the shipped configurations:

```bash
python -c "from sap3d import bench; print('\n'.join(c.case_id for c in bench.iter_cases('Edit3D-Bench/data')))" \
    > cases_300.txt
sap3d-bench --bench-root Edit3D-Bench/data --adapters adapters --out runs/local --cases cases_300.txt

grep -v 73fc1424 global_edits/cases.txt > global_local.txt     # 10 global cases, default configuration
grep 73fc1424 global_edits/cases.txt > global_uniform.txt      # the 2 wire-mesh pumpkins, uniform strength
sap3d-bench --bench-root global_edits --adapters adapters --out runs/global --cases global_local.txt --global
sap3d-bench --bench-root global_edits --adapters adapters --out runs/global_uniform --cases global_uniform.txt \
            --global --config configs/edit_global_uniform.yaml
```

`summary.md` holds the means. The paper's edit-fidelity columns are the color-fitted ones for local edits and
the generated ones for global edits. Renders and the benchmark's Chamfer distance vary slightly from run to run.

## Limitations

- Assets are glTF/GLB files with static meshes; texturing needs a source that is one mesh with one base-color
  texture.
- An edited view without transparency needs a near-white background.
- The metrics see the edited view only; the far side of an edit is not measured.
- Retraining does not reproduce a downloaded adapter exactly: the training renders and samples are random.
- Blackwell GPUs are untested.

## Tests

```bash
python tests/run_all.py
```

CPU only; run from the repository root.

## Citation

```bibtex
@article{levin2026sap3d,
  title   = {SAP3D: Spatially Adaptive Personalization for Consistent 3D Generative Editing},
  author  = {Levin, Itan and Tevet, Guy and Bermano, Amit H.},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

The Tencent Hunyuan 3D 2.1 Community License Agreement (`LICENSE`, `NOTICE`). `third_party/hy3dshape` is
a modified copy of Tencent's package; its changes are listed in `third_party/hy3dshape/MODIFICATIONS.md`.
`third_party/hy3dpaint` is an unmodified subset of Tencent's paint package. `third_party/edit3d_bench` is
the Edit3D-Bench evaluation code from the VoxHammer repository, unmodified, under its MIT license.
