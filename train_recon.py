import argparse
import math
import os
import csv
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection

from attention import IPAttnProcessor_mask2_0


# ------------- 数据集 -------------
class ImageFolder256(Dataset):
    def __init__(self, root: str):
        self.dataset = ImageFolder(
            root,
            transform=transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return {"pixel_values": img}


# ------------- 配置 -------------
@dataclass
class TrainConfig:
    model_path: str = "models/sd15"
    image_encoder_path: str = "models/clip-vit-large-patch14"
    train_data_dir: str = "ffhq256"
    output_dir: str = "outputs/recon_stage1"
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_train_steps: Optional[int] = None
    num_tokens: int = 4
    scale_img: float = 1.0
    scale_text: float = 0.0
    seed: int = 42
    log_interval: int = 100


# ------------- 工具 -------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_ip_tokens(image_encoder, pixel_values: torch.Tensor, num_tokens: int) -> torch.Tensor:
    # Resize to CLIP expected resolution (default 224) before encoding
    vision_size = getattr(image_encoder.config, "image_size", 224)
    if pixel_values.shape[-1] != vision_size or pixel_values.shape[-2] != vision_size:
        pixel_values = F.interpolate(pixel_values, size=(vision_size, vision_size), mode="bicubic", align_corners=False)
    vision_out = image_encoder(pixel_values, output_hidden_states=True)
    feats = vision_out.last_hidden_state  # (b, seq, dim)
    if feats.shape[1] < num_tokens:
        raise ValueError(f"image encoder tokens {feats.shape[1]} < num_tokens {num_tokens}")
    return feats[:, :num_tokens, :]


def replace_unet_attention(unet: UNet2DConditionModel, hidden_size: int, cross_attention_dim: int, cfg: TrainConfig):
    # 为每个注意力层单独设定 hidden_size/cross_attention_dim
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
            hs = hidden_size  # fallback

        # attn1 是自注意力，没有 cross，其他是 cross
        cross_dim = None if name.endswith("attn1.processor") else cross_attention_dim

        attn_procs[name] = IPAttnProcessor_mask2_0(
            hidden_size=hs,
            cross_attention_dim=cross_dim,
            num_tokens=cfg.num_tokens,
            scale_img=cfg.scale_img,
            scale_text=cfg.scale_text,
            use_mask=False,
            train_ip=True,
        )

    unet.set_attn_processor(attn_procs)


# ------------- 训练循环 -------------
def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # 模型
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(cfg.model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(cfg.model_path, subfolder="unet").to(device)

    # vision encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(device)
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # project image tokens to UNet cross_attention_dim (text dim)
    ip_proj = torch.nn.Linear(image_encoder.config.hidden_size, unet.config.cross_attention_dim).to(device)

    # 替换 attention processor
    replace_unet_attention(
        unet,
        hidden_size=unet.config.cross_attention_dim,
        cross_attention_dim=unet.config.cross_attention_dim,
        cfg=cfg,
    )
    unet.to(device)  # ensure new attention processors are on the same device

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    # 冻结不训练的模块
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    ip_proj.requires_grad_(True)

    # 仅训练 IP 分支参数 + 投影层
    train_params = []
    for _, module in unet.attn_processors.items():
        for n, p in module.named_parameters():
            if "to_k_ip" in n or "to_v_ip" in n:
                p.requires_grad_(True)
                train_params.append(p)
            else:
                p.requires_grad_(False)
    train_params += list(ip_proj.parameters())

    optimizer = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    dataset = ImageFolder256(cfg.train_data_dir)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    total_steps = cfg.max_train_steps or math.ceil(len(dataloader) / cfg.gradient_accumulation_steps) * cfg.num_epochs
    global_step = 0

    os.makedirs(cfg.output_dir, exist_ok=True)
    loss_log_path = os.path.join(cfg.output_dir, "loss_log.csv")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss"])

    for epoch in range(cfg.num_epochs):
        epoch_bar = tqdm(dataloader, desc=f"epoch {epoch+1}/{cfg.num_epochs}", leave=False)
        for batch in epoch_bar:
            pixel_values = batch["pixel_values"].to(device)
            # 归一化到 image encoder 需求
            pixel_values_clip = (pixel_values * 0.5 + 0.5 - image_mean.to(device)) / image_std.to(device)

            with torch.no_grad():
                # 文本：空提示
                text_inputs = tokenizer(
                    [""] * pixel_values.shape[0],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

                # IP tokens（CLIP 冻结）
                ip_tokens = get_ip_tokens(image_encoder, pixel_values_clip, cfg.num_tokens)

                # VAE encode
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # 拼接 [text | ip]，先将 IP tokens 投影到 text 维度
            ip_tokens = ip_proj(ip_tokens)
            encoder_hidden_states = torch.cat([text_embeddings, ip_tokens], dim=1)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss_noise = F.mse_loss(model_pred, noise)
            loss = loss_noise

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

        # 每个 epoch 保存一次（包含 UNet 和 ip_proj）
        save_path = os.path.join(cfg.output_dir, f"unet_ip_stage1_epoch{epoch+1}.bin")
        torch.save({"unet": unet.state_dict(), "ip_proj": ip_proj.state_dict()}, save_path)
        print(f"Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/sd15", help="Stable Diffusion 1.5 checkpoint path")
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="models/clip-vit-large-patch14",
        help="CLIP image encoder path (downloaded)",
    )
    parser.add_argument("--train_data_dir", type=str, default="ffhq256", help="FFHQ256 root directory")
    parser.add_argument("--output_dir", type=str, default="outputs/recon_stage1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--scale_img", type=float, default=1.0)
    parser.add_argument("--scale_text", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between writing loss to CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_path=args.model_path,
        image_encoder_path=args.image_encoder_path,
        train_data_dir=args.train_data_dir,
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
    )
    train(cfg)
