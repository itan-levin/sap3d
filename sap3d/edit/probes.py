"""The two probes that nominate the latent tokens an edit region covers.

The 3D probe decodes the predicted clean latent through the shape VAE and reads the geometry decoder's
cross-attention from query points inside the region to the latent tokens. The 2D probe reads the DiT's
cross-attention from latent tokens to the conditioning-image patches under the 2D mask, captured during the
step's own forward pass, from chosen (layer, head) pairs.
"""

import numpy as np
import torch

from hy3dshape.models.autoencoders.attention_processors import scaled_dot_product_attention

from .knee import knee_region


class HeadAttentionSum:
    """Capture storage summing the cross-attention weights of chosen (layer, head) pairs over one forward.

    Layers are counted in the order their cross-attention runs, which is block order. The sum is kept in
    the weights' own precision, the first contribution assigned and later ones added in place.
    """

    enabled = True

    def __init__(self, heads):
        self.heads = tuple((int(layer), int(head)) for layer, head in heads)
        self._layer = 0
        self._sum = None
        self._count = 0

    def store(self, weights: torch.Tensor):
        layer = self._layer
        self._layer += 1
        head_indices = [head for (l, head) in self.heads if l == layer]
        if not head_indices:
            return
        contribution = weights[:, head_indices, :, :].detach().sum(dim=1)
        if self._sum is None:
            self._sum = contribution
            self._count = len(head_indices)
        else:
            self._sum += contribution
            self._count += len(head_indices)

    def mean(self) -> torch.Tensor:
        """[batch, latent tokens + 1, image tokens] mean over the captured (layer, head) pairs."""
        if self._count != len(self.heads):
            raise ValueError(f"cross-attention was captured for {self._count} of the heads {self.heads}")
        return self._sum / self._count


def patch_region(storage: HeadAttentionSum, patches: list, knee) -> set:
    """Latent tokens (0-based) nominated by the conditional half's attention to the given 1-based patches."""
    xattn = storage.mean()[0, 1:, :].cpu().numpy()
    return knee_region(xattn[:, patches].T, knee.factor, knee.width, knee.sensitivity)


class GeoDecoderCaptureProcessor:
    """The geometry decoder's attention processor, also keeping softmax(q kᵀ) averaged over heads.

    The returned attention output is the default processor's, so decoding is unchanged. Weights are
    computed in float32 and kept as float16, [batch, queries, latent tokens].
    """

    def __init__(self):
        self.captured = None

    def __call__(self, attn, q, k, v):
        out = scaled_dot_product_attention(q, k, v)
        scale = q.shape[-1] ** -0.5
        weights = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1)
        self.captured = weights.mean(dim=1).to(torch.float16)
        return out


def decoder_attention(vae, x1_hat: torch.Tensor, query_coords: np.ndarray) -> np.ndarray:
    """[queries, latent tokens] float16 decoder cross-attention for model-frame query points."""
    parameter = next(vae.parameters())
    queries = torch.as_tensor(np.asarray(query_coords), dtype=torch.float32, device=parameter.device)
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
    geo = vae.geo_decoder
    previous = geo.cross_attn_decoder.attn.attention.attn_processor
    capture = GeoDecoderCaptureProcessor()
    geo.set_cross_attention_processor(capture)
    try:
        with torch.no_grad():
            latents = vae.decode((1.0 / vae.scale_factor) * x1_hat.to(device=parameter.device,
                                                                       dtype=parameter.dtype))
            geo(queries=queries, latents=latents)
    finally:
        geo.set_cross_attention_processor(previous)
    return capture.captured[0].cpu().numpy()


def point_region(vae, x1_hat: torch.Tensor, query_coords: np.ndarray, knee) -> set:
    return knee_region(decoder_attention(vae, x1_hat, query_coords), knee.factor, knee.width, knee.sensitivity)
