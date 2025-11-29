import argparse
import os
import csv
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection

from attention import IPAttnProcessor_mask2_0

try:
    from lpips import LPIPS
except ImportError:
    LPIPS = None


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img) * 2.0 - 1.0


class DefectDataset(Dataset):
    def __init__(self, root: str, size: int = 256):
        self.def_dir = os.path.join(root, "defects")
        self.clean_dir = os.path.join(root, "clean")
        self.mask_dir = os.path.join(root, "masks")
        self.size = size

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        self.files = [f for f in os.listdir(self.def_dir) if os.path.splitext(f)[1].lower() in exts]
        self.files.sort()

        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        def_img = Image.open(os.path.join(self.def_dir, name)).convert("RGB")
        clean_img = Image.open(os.path.join(self.clean_dir, name)).convert("RGB")
        mask_img = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        def_img = self.transform(def_img)
        clean_img = self.transform(clean_img)
        mask_img = self.transform(mask_img)

        return {
            "defect": pil_to_tensor(def_img),
            "clean": pil_to_tensor(clean_img),
            "mask": transforms.ToTensor()(mask_img),
            "name": name,
        }


@dataclass
class TrainConfig:
    model_path: str = "models/sd15"
    image_encoder_path: str = "models/clip-vit-large-patch14"
    defect_data_dir: str = "ffhq256_defect_scribble"
    output_dir: str = "outputs/mask_stage2"
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 2
    gradient_accumulation_steps: int = 1
    max_train_steps: Optional[int] = None
    num_tokens: int = 4
    scale_img: float = 1.0
    scale_text: float = 0.0
    seed: int = 42
    log_interval: int = 100
    lpips_weight: float = 0.0
    l1_weight: float = 0.0
    stage1_ckpt: str = "outputs/recon_stage1_run1/unet_ip_stage1_epoch1.bin"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_ip_tokens(image_encoder, pixel_values: torch.Tensor, num_tokens: int) -> torch.Tensor:
    vision_size = getattr(image_encoder.config, "image_size", 224)
    if pixel_values.shape[-1] != vision_size or pixel_values.shape[-2] != vision_size:
        pixel_values = F.interpolate(pixel_values, size=(vision_size, vision_size), mode="bicubic", align_corners=False)
    vision_out = image_encoder(pixel_values, output_hidden_states=True)
    feats = vision_out.last_hidden_state  # (b, seq, dim)
    if feats.shape[1] < num_tokens:
        raise ValueError(f"image encoder tokens {feats.shape[1]} < num_tokens {num_tokens}")
    return feats[:, :num_tokens, :]


def replace_unet_attention(unet: UNet2DConditionModel, cross_attention_dim: int, cfg: TrainConfig, train_mask: bool = True):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            hs = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hs = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hs = unet.config.block_out_channels[block_id]
        else:
            hs = cross_attention_dim

        cross_dim = None if name.endswith("attn1.processor") else cross_attention_dim

        attn_procs[name] = IPAttnProcessor_mask2_0(
            hidden_size=hs,
            cross_attention_dim=cross_dim,
            num_tokens=cfg.num_tokens,
            scale_img=cfg.scale_img,
            scale_text=cfg.scale_text,
            use_mask=train_mask,
            train_ip=not train_mask,
        )
    unet.set_attn_processor(attn_procs)


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(cfg.model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(cfg.model_path, subfolder="unet").to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(device)
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # replace attention for mask training
    replace_unet_attention(unet, cross_attention_dim=unet.config.cross_attention_dim, cfg=cfg, train_mask=True)
    unet.to(device)

    # projection layer (load from stage1)
    ip_proj = torch.nn.Linear(image_encoder.config.hidden_size, unet.config.cross_attention_dim).to(device)

    # load stage1 weights
    ckpt = torch.load(cfg.stage1_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "unet" in ckpt and "ip_proj" in ckpt:
        unet.load_state_dict(ckpt["unet"], strict=False)
        ip_proj.load_state_dict(ckpt["ip_proj"], strict=False)
    else:
        unet.load_state_dict(ckpt, strict=False)

    # freeze everything except mask branch
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    ip_proj.requires_grad_(False)
    for _, module in unet.attn_processors.items():
        module.freeze_ip()
        module.use_mask = True
        for param in module.parameters():
            param.requires_grad_(False)
        # unfreeze mask params
        for name, param in module.named_parameters():
            if "to_k_ip_mask" in name or "to_v_ip_mask" in name or "to_out_ip_mask" in name:
                param.requires_grad_(True)

    train_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    dataset = DefectDataset(cfg.defect_data_dir, size=256)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    lpips_fn = None
    if cfg.lpips_weight > 0:
        if LPIPS is None:
            raise ImportError("lpips is not installed but lpips_weight > 0")
        lpips_fn = LPIPS(net="vgg").to(device)

    os.makedirs(cfg.output_dir, exist_ok=True)
    loss_log_path = os.path.join(cfg.output_dir, "loss_log.csv")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss"])

    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_bar = tqdm(dataloader, desc=f"epoch {epoch+1}/{cfg.num_epochs}", leave=False)
        for batch in epoch_bar:
            defect = batch["defect"].to(device)
            clean = batch["clean"].to(device)
            mask = batch["mask"].to(device)  # 0-1

            # for image encoder: normalize to CLIP space
            defect_clip = (defect * 0.5 + 0.5 - image_mean.to(device)) / image_std.to(device)

            with torch.no_grad():
                text_inputs = tokenizer(
                    [""] * defect.shape[0],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
                ip_tokens = get_ip_tokens(image_encoder, defect_clip, cfg.num_tokens)
                latents = vae.encode(clean).latent_dist.sample() * 0.18215

            ip_tokens_proj = ip_proj(ip_tokens)
            encoder_hidden_states = torch.cat([text_embeddings, ip_tokens_proj], dim=1)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

            # optional x0 reconstruction for mask loss
            if (cfg.lpips_weight > 0 or cfg.l1_weight > 0) and scheduler.alphas_cumprod is not None:
                alphas = scheduler.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
                sqrt_alpha = torch.sqrt(alphas)
                sqrt_one_minus = torch.sqrt(1 - alphas)
                pred_x0 = (noisy_latents - sqrt_one_minus * model_pred) / sqrt_alpha
                with torch.no_grad():
                    recon = vae.decode(pred_x0 / 0.18215).sample
                if cfg.lpips_weight > 0 and lpips_fn is not None:
                    loss_lpips = lpips_fn(recon, clean).mean()
                    loss = loss + cfg.lpips_weight * loss_lpips
                if cfg.l1_weight > 0:
                    loss_l1 = F.l1_loss(recon * mask, clean * mask)
                    loss = loss + cfg.l1_weight * loss_l1

            loss.backward()

            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_bar.set_postfix(loss=float(loss.detach()))

            if global_step % cfg.log_interval == 0:
                with open(loss_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch + 1, float(loss.detach())])

            global_step += 1
            if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                break

        save_path = os.path.join(cfg.output_dir, f"unet_ip_mask_epoch{epoch+1}.bin")
        torch.save({"unet": unet.state_dict(), "ip_proj": ip_proj.state_dict()}, save_path)
        print(f"Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/sd15")
    parser.add_argument("--image_encoder_path", type=str, default="models/clip-vit-large-patch14")
    parser.add_argument("--defect_data_dir", type=str, default="ffhq256_defect_scribble")
    parser.add_argument("--output_dir", type=str, default="outputs/mask_stage2")
    parser.add_argument("--stage1_ckpt", type=str, default="outputs/recon_stage1_run1/unet_ip_stage1_epoch1.bin")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--scale_img", type=float, default=1.0)
    parser.add_argument("--scale_text", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lpips_weight", type=float, default=0.0)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_path=args.model_path,
        image_encoder_path=args.image_encoder_path,
        defect_data_dir=args.defect_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        num_tokens=args.num_tokens,
        scale_img=args.scale_img,
        scale_text=args.scale_text,
        seed=args.seed,
        log_interval=args.log_interval,
        lpips_weight=args.lpips_weight,
        l1_weight=args.l1_weight,
        stage1_ckpt=args.stage1_ckpt,
    )
    train(cfg)
