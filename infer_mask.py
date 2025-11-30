import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection

from attention import IPAttnProcessor_mask2_0


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_image(path: str, size: int = 256) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image)


def get_ip_tokens(image_encoder, pixel_values: torch.Tensor, num_tokens: int) -> torch.Tensor:
    vision_size = getattr(image_encoder.config, "image_size", 224)
    if pixel_values.shape[-1] != vision_size or pixel_values.shape[-2] != vision_size:
        pixel_values = F.interpolate(pixel_values, size=(vision_size, vision_size), mode="bicubic", align_corners=False)
    vision_out = image_encoder(pixel_values, output_hidden_states=True)
    feats = vision_out.last_hidden_state  # (b, seq, dim)
    if feats.shape[1] < num_tokens:
        raise ValueError(f"image encoder tokens {feats.shape[1]} < num_tokens {num_tokens}")
    return feats[:, :num_tokens, :]


def replace_unet_attention(unet: UNet2DConditionModel, cross_attention_dim: int, num_tokens: int, scale_img: float, scale_text: float, use_mask: bool):
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
            num_tokens=num_tokens,
            scale_img=scale_img,
            scale_text=scale_text,
            use_mask=use_mask,
            train_ip=False,
        )
    unet.set_attn_processor(attn_procs)


def save_image(tensor: torch.Tensor, path: str):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    tensor = tensor.mul(255).byte().cpu()
    pil = transforms.ToPILImage()(tensor.squeeze(0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil.save(path)
    print(f"Saved: {path}")


def infer(
    model_path: str,
    image_encoder_path: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    input_image: str,
    mask_image: Optional[str],
    output_path: str,
    num_tokens: int = 16,
    num_steps: int = 50,
    seed: int = 42,
    scale_img: float = 1.0,
    scale_text: float = 0.0,
    mask_dilate: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device)
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)

    # replace attention with mask enabled
    cross_dim = unet.config.cross_attention_dim
    replace_unet_attention(
        unet, cross_attention_dim=cross_dim, num_tokens=num_tokens, scale_img=scale_img, scale_text=scale_text, use_mask=True
    )
    unet.to(device)

    # projection
    ip_proj = torch.nn.Linear(image_encoder.config.hidden_size, cross_dim).to(device)

    # load stage1 + stage2
    ckpt1 = torch.load(stage1_ckpt, map_location="cpu")
    if isinstance(ckpt1, dict) and "unet" in ckpt1 and "ip_proj" in ckpt1:
        unet.load_state_dict(ckpt1["unet"], strict=False)
        ip_proj.load_state_dict(ckpt1["ip_proj"], strict=False)
    else:
        unet.load_state_dict(ckpt1, strict=False)
    ckpt2 = torch.load(stage2_ckpt, map_location="cpu")
    if isinstance(ckpt2, dict) and "unet" in ckpt2:
        unet.load_state_dict(ckpt2["unet"], strict=False)
    # ip_proj from stage1 remains

    # prepare inputs
    pixel_values = load_image(input_image).unsqueeze(0).to(device)
    pixel_values_clip = (pixel_values * 0.5 + 0.5 - image_mean) / image_std

    mask_tensor = None
    if mask_image:
        mask_img = Image.open(mask_image).convert("L")
        mask_tensor = transforms.ToTensor()(mask_img)
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1,1,H,W)
        if mask_dilate > 0:
            kernel = torch.ones((1, 1, mask_dilate, mask_dilate), device=mask_tensor.device)
            padding = mask_dilate // 2
            mask_tensor = torch.clamp(torch.nn.functional.conv2d(mask_tensor, kernel, padding=padding), 0, 1)
        mask_tensor = F.interpolate(mask_tensor, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        text_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(text_inputs.input_ids)[0]
        ip_tokens = get_ip_tokens(image_encoder, pixel_values_clip, num_tokens)
        ip_tokens = ip_proj(ip_tokens)
        encoder_hidden_states = torch.cat([text_embeddings, ip_tokens], dim=1)

        init_latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

    # DDIM scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(num_steps, device=device)

    # start from pure noise
    latents = torch.randn(
        (1, unet.config.in_channels, pixel_values.shape[-2] // 8, pixel_values.shape[-1] // 8),
        device=device,
    )
    timesteps = scheduler.timesteps

    with torch.no_grad():
        for t in timesteps:
            model_output = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            latents = scheduler.step(model_output, t, latents).prev_sample

        images = vae.decode(latents / 0.18215).sample

    # optional mask: blend input and output to visualize repaired region
    if mask_tensor is not None:
        images = images * mask_tensor + pixel_values * (1 - mask_tensor)

    save_image(images, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with mask branch")
    parser.add_argument("--model_path", type=str, default="models/sd15")
    parser.add_argument("--image_encoder_path", type=str, default="models/clip-vit-large-patch14")
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Path to stage1 (recon) checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to stage2 (mask) checkpoint")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, default=None, help="Optional mask for visualization blending")
    parser.add_argument("--output_path", type=str, default="outputs/mask_eval/out.png")
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale_img", type=float, default=1.0)
    parser.add_argument("--scale_text", type=float, default=0.0)
    parser.add_argument("--mask_dilate", type=int, default=0, help="Optional dilation for provided mask in blending")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(
        model_path=args.model_path,
        image_encoder_path=args.image_encoder_path,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        input_image=args.input_image,
        mask_image=args.mask_image,
        output_path=args.output_path,
        num_tokens=args.num_tokens,
        num_steps=args.num_steps,
        seed=args.seed,
        scale_img=args.scale_img,
        scale_text=args.scale_text,
        mask_dilate=args.mask_dilate,
    )
