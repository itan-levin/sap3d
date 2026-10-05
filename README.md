# SAP3D

Localized editing of 3D assets with per-object personalized adapters on top of Tencent's
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape model. Given a source mesh, its
adapter, an edited view of the source and a 3D region, the edit stage regenerates the region to match
the edited view and keeps the rest of the object.

The repository is organized as independent stages: personalize (asset to adapter), edit, texture and
eval, plus a batch runner for Edit3D-Bench.

## Install

Python 3.10 with a CUDA GPU. Two pinned environments are provided; each brings the CUDA compiler that
`build_extensions.sh` needs:

```bash
conda env create -f environment.yml            # PyTorch 2.5.1, CUDA 12.4
conda activate sap3d
bash scripts/build_extensions.sh               # diso, custom_rasterizer, the mesh painter
pip install -e .
```

Use a clone of this repository with an editable install (or run from the repository root with
`PYTHONPATH=$PWD:$PWD/third_party/hy3dshape`): the texture and eval stages use the vendored paint model
(`third_party/hy3dpaint`) and the vendored Edit3D-Bench scorer (`third_party/edit3d_bench`) by path. The
paint model also needs `RealESRGAN_x4plus.pth` from the Real-ESRGAN v0.1.0 release at
`third_party/hy3dpaint/ckpt/RealESRGAN_x4plus.pth` (or the path set as `realesrgan_ckpt` in a texture
configuration).

or `environment-torch28.yml` for PyTorch 2.8.0 (CUDA 12.8), which also pins `setuptools<81`.

The shape model weights (`tencent/Hunyuan3D-2.1`) are downloaded from Hugging Face on first use into
`~/.cache/hy3dgen` (override with `HY3DGEN_MODELS`); the paint model (`hunyuan3d-paintpbr-v2-1` from the
same repository, `facebook/dinov2-giant`), the evaluation encoders (`facebook/dinov2-base`,
`openai/clip-vit-base-patch32`, LPIPS AlexNet) and the background remover download on first use.

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

## Texture

```bash
sap3d-texture --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2 \
              --mesh outputs/sorting_bus_p2/edit.glb --out outputs/sorting_bus_p2/texture
```

or, for any asset, `--mesh edit.glb --source model.glb --edited-view edited.png --region region.yaml`
plus the edited view's camera (`--camera-meta meta.json --view K`, `--camera camera.json` or
`--orbit ELEVATION AZIMUTH`) and optionally `--mask`.

- **Composite mode** (default, local edits). The edited mesh is decimated to 40,000 faces and given its own
  UV atlas. Every texel copies the source asset's color from the closest point on the source surface
  (rejected when the surface normals disagree). The paint model paints the whole mesh from the edited view,
  recomposed to fill 90% of a 512 px frame on mid-gray, and its texels replace the copied ones inside the
  region, in a 0.02-unit shell just outside it, and on surface outside it whose copied color came from inside
  it; a 0.01-unit band beyond the shell blends painted into copied color. The source asset must be one
  mesh with one base-color texture.
- **Color fit** (composite mode). The painted region's CIELAB lightness and chroma are adjusted until a
  render through the edited view's camera matches the edited view inside the 2D mask (up to six rounds; the
  uncorrected texture competes and wins when nothing improves on it). Writes `composite_color_fit.glb`
  beside `composite.glb`.
- **Generated mode** (`--mode generated`, global edits) publishes the painted texture alone.

The output GLBs declare what Edit3D-Bench's own GLBs declare (positions and UVs, one PNG base-color texture,
roughness 1, single-sided), so the benchmark renderer shades them like the source objects.
`configs/texture.yaml` lists the settings.

## Eval

```bash
sap3d-eval --texture-dir outputs/sorting_bus_p2/texture --out outputs/sorting_bus_p2/eval \
           --bench-root Edit3D-Bench/data --case GSO/SORTING_BUS/prompt_2
```

- **Preservation**: exact point-to-surface distance to the source mesh, in its units (200,000 surface
  samples per mesh, fixed seed, symmetric mean). Composite-mode textures are compared outside the region,
  on the composite and on the raw edited mesh; generated-mode textures over the whole surface.
- **Edit fidelity**: a render through the edited view's camera against the edited view, inside the 2D
  mask's square bounding box with everything outside the mask set to gray; 1 − cosine of DINOv2-base class
  tokens and of CLIP ViT-B/32 image embeddings. Scored on the composite and on the color-fitted mesh (or the
  generated mesh). The encoders run on the CPU for composite-mode textures and on `cuda:0` for generated
  ones, the devices behind the published numbers (`--fidelity-device` overrides; the two differ by up to
  about 1e-4 for DINOv2 and 3e-4 for CLIP).
- **Benchmark metrics** (benchmark cases, composite mode): the Edit3D-Bench scorer, unmodified, on the
  composite: Chamfer distance, PSNR, SSIM, LPIPS, DINO-I and CLIP-T at 512 px.

In the paper's main table, the benchmark metrics and preservation are those of the composite before the
color fit, and edit fidelity is that of the color-fitted mesh.

Blender renders are not exactly repeatable: two renders of one mesh differ on a few dozen of a million
pixels by one 8-bit level, so render-based numbers repeat to about 1e-4 (fidelity) or better. The scorer
samples its Chamfer points without a seed, so that metric varies at the 1e-5 level between runs.

## Benchmark runs

```bash
sap3d-bench --bench-root Edit3D-Bench/data --adapters adapters/ --out runs/bench --cases cases.txt
```

`cases.txt` holds one case id per line. The runner edits, textures, color-fits and evaluates every case,
writing `runs/bench/cases/<case id>/{edit,texture,eval}/` and `case.json`, then `summary.json`,
`summary.csv` and `summary.md` over every finished case under the output folder (means are taken separately
for local and global edits; use one output folder per configuration). A rerun skips finished stages whose
settings and inputs match, redoes stages whose inputs changed, and refuses stages made with other settings;
`--overwrite` redoes every stage of every listed case. A failing case, including a color fit that fails,
stops the run; `--skip-cases` leaves cases out. `--global` runs whole-object edits (generated texture, whole-surface
preservation, no benchmark metrics), with the region and mask files of each case folder or with
`--config configs/edit_global_uniform.yaml`.

## Tests

```bash
python tests/run_all.py
```

CPU only; run from the repository root.

## License

The Tencent Hunyuan 3D 2.1 Community License Agreement (`LICENSE`, `NOTICE`). `third_party/hy3dshape` is
a modified copy of Tencent's package; its changes are listed in `third_party/hy3dshape/MODIFICATIONS.md`.
`third_party/hy3dpaint` is an unmodified subset of Tencent's paint package. `third_party/edit3d_bench` is
the Edit3D-Bench evaluation code from the VoxHammer repository, unmodified, under its MIT license.
