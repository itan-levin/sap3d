"""Fit a per-object adapter: LoRA on the feed-forward layers of the Hunyuan3D-2.1 shape DiT, trained with the
flow-matching objective on one asset's dataset (see `dataset`).

The DiT with its adapter, the shape VAE and the image encoder are held in fp16, and so are the adapter's
optimizer state and gradients; the forward pass runs under bf16 autocast (fp16 for the VAE encoding, which
draws a sample from the posterior). Every `validation.every` steps the adapter is saved to
`<out>/steps/step_<N>/` and validated (see `validate`); validation also runs once before training.
`<out>/training.json` records the settings, every step's loss and every validation, and the chosen step is
copied to `<out>/adapter/`.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import PersonalizeConfig, TrainConfig, personalize_to_dict
from . import select as selection
from .data import SampleStream
from .validate import ValidationStream, check_extractor, validate

DIT_KWARGS = dict(input_size=4096, in_channels=64, hidden_size=2048, context_dim=1024, depth=21, num_heads=16,
                  qk_norm=True, text_len=1370, with_decoupled_ca=False, use_attention_pooling=False,
                  qk_norm_type="rms", qkv_bias=False, use_pos_emb=False, num_moe_layers=6, num_experts=8,
                  moe_top_k=2)
IMAGE_ENCODER = {"type": "DinoImageEncoder",
                 "kwargs": {"version": "facebook/dinov2-large", "image_size": 518, "use_cls_token": True}}
# The edit stage expects exactly these: fc1/fc2 of the block MLPs and net.0.proj/net.2 of the MoE experts.
LORA_TARGETS = ["fc1", "fc2", "net.0.proj", "net.2"]
Z_SCALE_FACTOR = 1.0039506158752403


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Models:
    dit: torch.nn.Module
    encoder: torch.nn.Module
    vae: torch.nn.Module
    transport: object
    pipeline: object


def build_models(cfg: TrainConfig, model_path: str = "tencent/Hunyuan3D-2.1") -> Models:
    """The models in training order: the LoRA initialization depends on the random state left by the DiT and
    the image encoder, and the pipeline's constructor moves everything to CUDA in fp16."""
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.models.conditioner import SingleImageEncoder
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from hy3dshape.models.diffusion.transport import create_transport
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.preprocessors import ImageProcessorV2
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig, get_peft_model

    seed_everything(cfg.seed)
    transport = create_transport(path_type="Linear", prediction="velocity")
    dit = HunYuanDiTPlain.from_pretrained(model_path, use_safetensors=False, variant="fp16", **DIT_KWARGS)
    encoder = SingleImageEncoder(main_image_encoder=IMAGE_ENCODER, drop_ratio=0.0)
    dit = get_peft_model(dit, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, target_modules=LORA_TARGETS))
    vae = ShapeVAE.from_pretrained(model_path, use_safetensors=False, variant="fp16",
                                   pc_size=cfg.points_per_sample, pc_sharpedge_size=0)
    vae.eval()
    vae.requires_grad_(False)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(
        vae=vae, model=dit, scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        conditioner=encoder, image_processor=ImageProcessorV2())
    return Models(dit=dit, encoder=encoder, vae=vae, transport=transport, pipeline=pipeline)


def make_optimizer(models: Models, cfg: TrainConfig):
    from hy3dshape.utils.trainings.lr_scheduler import LambdaWarmUpCosineFactorScheduler

    parameters = list(models.dit.parameters())
    optimizer = torch.optim.AdamW([{"params": parameters, "lr": cfg.learning_rate}], lr=cfg.learning_rate,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
                                  weight_decay=cfg.weight_decay)
    factor = LambdaWarmUpCosineFactorScheduler(warm_up_steps=cfg.warmup_steps, f_min=cfg.lr_factor_min,
                                               f_max=cfg.lr_factor_max, f_start=cfg.lr_factor_start,
                                               max_decay_steps=cfg.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor.schedule)
    return optimizer, scheduler


def to_device(batch: dict) -> dict:
    return {key: batch[key].cuda() for key in ("image", "mask", "surface")}


def training_loss(models: Models, batch: dict) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        contexts = models.encoder(image=batch["image"], text=None, mask=batch["mask"])
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            latents = models.vae.encode(batch["surface"], sample_posterior=True)
            latents = Z_SCALE_FACTOR * latents
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return models.transport.training_losses(models.dit, latents, dict(contexts=contexts))["loss"].mean()


def train_step(models: Models, optimizer, scheduler, batch: dict, clip_norm: float) -> float:
    loss = training_loss(models, batch)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], clip_norm)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def set_training(models: Models, training: bool) -> None:
    models.dit.train(training)
    models.encoder.train(training)


def run(cfg: PersonalizeConfig, data_dir, out_dir, model_path: str = "tencent/Hunyuan3D-2.1") -> dict:
    import peft

    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(parents=True)
    train_cfg, val_cfg = cfg.train, cfg.validation
    train_stream = SampleStream(data_dir, train_cfg, seed=train_cfg.seed)
    val_stream = ValidationStream(SampleStream(data_dir, train_cfg, seed=train_cfg.seed), train_cfg.batch_size,
                                  seed=train_cfg.seed)

    models = build_models(train_cfg, model_path)
    optimizer, scheduler = make_optimizer(models, train_cfg)
    record = {"settings": personalize_to_dict(cfg), "model": model_path, "dataset": str(data_dir),
              "renders": len(train_stream.paths),
              "extractor": check_extractor(models.vae, val_cfg.octree_resolution, "cuda"),
              "environment": {"torch": torch.__version__, "peft": peft.__version__,
                              "gpu": torch.cuda.get_device_name(0)},
              "loss": [], "validations": []}
    path = out_dir / "training.json"

    def validation(step: int) -> None:
        set_training(models, False)
        batch, item = val_stream.next()
        record["validations"].append({"step": step, **validate(models.pipeline, batch, item, val_cfg)})
        record["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        selection.write_record(path, record)

    validation(0)
    seed_everything(train_cfg.seed)
    set_training(models, True)
    for step in range(1, train_cfg.steps + 1):
        batch = to_device(train_stream.batch(train_cfg.batch_size))
        record["loss"].append(train_step(models, optimizer, scheduler, batch, train_cfg.grad_clip_norm))
        if step % val_cfg.every == 0:
            models.dit.save_pretrained(str(steps_dir / f"step_{step}"))
            validation(step)
            set_training(models, True)
    return selection.select(out_dir)
