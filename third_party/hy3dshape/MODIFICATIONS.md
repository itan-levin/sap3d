# Changes to Tencent's hy3dshape package

`hy3dshape/` is Tencent's Hunyuan3D-2.1 `hy3dshape/hy3dshape` package. Every file not listed here is
unchanged. Each changed or added file carries a one-line notice at its top.

## Changed

| file | change |
|---|---|
| `models/denoisers/hunyuandit.py` | `CrossAttention.forward` hands its attention weights to a capture storage when one is installed (`utils/attention_capture.py`); the attention output is unchanged. |
| `models/denoisers/moe_layers.py` | During MoE inference each routed expert runs with its rows' batch and flat token indices published (and cleared afterwards, even when the expert raises), so an adapter inside an expert can apply a per-token strength. `moe_infer` takes the batch indices as a fourth argument. |
| `pipelines.py` | `Hunyuan3DDiTFlowMatchingPipeline.__call__` accepts `lora_scales`, one adapter strength per batch element, cleared after each forward pass even when it raises. `mc_algo` defaults to `None` in `_export` and `enable_flashvdm`, so the surface extractor is resolved by the VAE. `set_surface_extractor` uses the lookup helper and no longer logs a deprecation message. |
| `models/autoencoders/surface_extractors.py` | The DMC extractor places vertices in the decode bounds (as marching cubes does) instead of re-centering the mesh, honors `mc_level`, raises on a field with no surface, and fails at construction when `diso` is unusable. Marching cubes scales by the decoded grid's own cell count and no longer takes `octree_resolution`; `_compute_box_stat` became `_compute_box`, returning only the box. Adds `surface_extractor_from_name` and `default_surface_extractor`. |
| `models/autoencoders/model.py` | The constructor no longer defaults `surface_extractor` to marching cubes; `VectsetVAE.resolve_surface_extractor` picks one per decode: a named extractor (`mc_algo`) wins, then an assigned one, then DMC for a CUDA grid and marching cubes otherwise. Upstream used marching cubes unless FlashVDM decoding was enabled. `enable_flashvdm_decoder` defaults `mc_algo` to `None` (upstream `'dmc'`) and no longer resets the extractor to marching cubes when disabling. |
| `models/autoencoders/__init__.py` | Exports the two lookup helpers. |

## Added

| file | content |
|---|---|
| `models/per_sample_lora.py` | `PerSampleLoRALinear`, which multiplies a PEFT LoRA layer's adapter term by a strength per batch element or per token, the contexts carrying those strengths, and the conversion from a `PeftModel`. |
| `utils/attention_capture.py` | The capture-storage context used by the cross-attention hook. |
