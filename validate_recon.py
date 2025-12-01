import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from ip_adapter.ip_adapter import IPAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Validate IP-Adapter reconstruction on a few images.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to SD base model (e.g., models/sd15).")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="Path to CLIP image encoder.")
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        required=True,
        help="Path to trained ip_adapter weights (.pth/.safetensors) or Accelerate checkpoint dir/.bin to be converted.",
    )
    parser.add_argument("--input", type=str, required=True, help="Image file or directory to reconstruct.")
    parser.add_argument("--output_dir", type=str, default="outputs/recon_eval", help="Where to save reconstructions.")
    parser.add_argument("--num_tokens", type=int, default=16, help="Number of image tokens used during training.")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps for inference.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale; use 1.0 to disable text guidance.")
    parser.add_argument("--scale", type=float, default=1.0, help="Image prompt scale inside IP-Adapter.")
    parser.add_argument("--limit", type=int, default=4, help="Max number of images to process (for quick check).")
    parser.add_argument("--device", type=str, default=None, help="Device like cuda:0 or cpu; auto-detect if not set.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic outputs.")
    return parser.parse_args()


def list_images(path: Path):
    if path.is_file():
        return [path]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])
    return imgs


def extract_ip_weights(acc_ckpt_path: Path, save_dir: Path) -> Path:
    state = torch.load(acc_ckpt_path, map_location="cpu")
    image_proj = {
        k.replace("ip_adapter.image_proj_model.", ""): v
        for k, v in state.items()
        if k.startswith("ip_adapter.image_proj_model.")
    }
    ip_adapter = {
        k.replace("ip_adapter.adapter_modules.", ""): v
        for k, v in state.items()
        if k.startswith("ip_adapter.adapter_modules.")
    }
    if len(image_proj) == 0 or len(ip_adapter) == 0:
        raise ValueError(f"Checkpoint {acc_ckpt_path} does not contain ip_adapter weights")

    base_name = acc_ckpt_path.parent.name if acc_ckpt_path.name == "pytorch_model.bin" else acc_ckpt_path.stem
    out_path = save_dir / f"{base_name}_ip_adapter_eval.pth"
    torch.save({"image_proj": image_proj, "ip_adapter": ip_adapter}, out_path)
    return out_path


def resolve_ip_ckpt(ip_ckpt_arg: str, output_dir: Path) -> Path:
    ckpt_path = Path(ip_ckpt_arg)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "pytorch_model.bin"
    if ckpt_path.suffix == ".bin":
        return extract_ip_weights(ckpt_path, output_dir)
    return ckpt_path


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    torch.backends.cudnn.benchmark = True

    images = list_images(Path(args.input))
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if len(images) == 0:
        raise ValueError(f"No images found at {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)
    ip_ckpt_path = resolve_ip_ckpt(args.ip_ckpt, Path(args.output_dir))

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None, feature_extractor=None
    ).to(device)

    ip = IPAdapter(
        pipe,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=str(ip_ckpt_path),
        device=device,
        num_tokens=args.num_tokens,
    )

    for img_path in images:
        pil_image = Image.open(img_path).convert("RGB")
        outputs = ip.generate(
            pil_image=pil_image,
            prompt="",
            negative_prompt="",
            scale=args.scale,
            num_samples=1,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
        )
        out_file = Path(args.output_dir) / f"{img_path.stem}_recon.png"
        outputs[0].save(out_file)
        print(f"saved {out_file}")


if __name__ == "__main__":
    main()
