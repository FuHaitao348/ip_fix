import argparse
import math
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# -------- 基础工具 --------
def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)


def value_noise(h, w, base=24, octaves=4, persistence=0.6):
    noise = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = base
    total = 0.0
    for _ in range(octaves):
        gh, gw = max(1, h // freq), max(1, w // freq)
        grid = np.random.rand(gh, gw).astype(np.float32)
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        noise += (np.asarray(img).astype(np.float32) / 255.0) * amp
        total += amp
        amp *= persistence
        freq = max(1, freq // 2)
    noise /= (total if total > 0 else 1.0)
    return noise


def irregularize(mask, jag=0.9, blur=1.0):
    h, w = mask.shape
    low = value_noise(h, w, base=32, octaves=3, persistence=0.55)
    m = mask * (0.6 + jag * low)
    m = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.asarray(m).astype(np.float32) / 255.0


def morph(mask, iters=1, mode="dilate"):
    im = Image.fromarray((mask * 255).astype(np.uint8))
    for _ in range(max(0, iters)):
        if mode == "dilate":
            im = im.filter(ImageFilter.MaxFilter(size=3))
        else:
            im = im.filter(ImageFilter.MinFilter(size=3))
    im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.asarray(im).astype(np.float32) / 255.0


def area_limit(mask, max_area):
    cur = float(mask.mean())
    if cur > 1e-6 and cur > max_area:
        mask = mask * (max_area / cur)
    return np.clip(mask, 0, 1)


# -------- 缺陷原型 --------
def mask_scratches(h, w, target_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    num = np.random.randint(8, 18)
    for _ in range(num):
        length = np.random.uniform(0.35, 0.95) * max(h, w)
        angle = np.random.uniform(-25, 25) + random.choice([0, 90, 180, 270])
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        segs = np.random.randint(8, 18)
        for s in range(segs):
            t0 = (s / segs) * length
            t1 = ((s + 0.5) / segs) * length
            x0 = int(cx + dx * t0 + np.random.uniform(-4, 4))
            y0 = int(cy + dy * t0 + np.random.uniform(-4, 4))
            x1 = int(cx + dx * t1 + np.random.uniform(-4, 4))
            y1 = int(cy + dy * t1 + np.random.uniform(-4, 4))
            th = np.random.uniform(0.8, 2.4)
            draw.line([(x0, y0), (x1, y1)], fill=255, width=int(max(1, th)))
    m = np.asarray(canvas).astype(np.float32) / 255.0
    m = morph(m, iters=1, mode="dilate")
    m = irregularize(m, jag=0.95, blur=0.9)
    m = area_limit(m, target_area)
    return m


def mask_dust_fiber(h, w, target_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
    m = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    n = max(40, int(target_area * h * w / 2500))
    for _ in range(n):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(1, 4)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        m[blob] = np.maximum(m[blob], 1.0)
    fib = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(fib)
    nf = max(10, int(n * 0.2))
    for _ in range(nf):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        l = np.random.randint(15, 50)
        a = np.random.uniform(0, 360)
        ex = int(cx + l * math.cos(math.radians(a)))
        ey = int(cy + l * math.sin(math.radians(a)))
        d.line([(cx, cy), (ex, ey)], fill=255, width=np.random.randint(1, 3))
    m = np.maximum(m, np.asarray(fib).astype(np.float32) / 255.0)
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.8))
    m = np.asarray(m).astype(np.float32) / 255.0
    m = irregularize(m, jag=0.9, blur=0.8)
    m = area_limit(m, target_area)
    return m


def mask_mottle(h, w, target_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
    base = value_noise(h, w, base=20, octaves=4, persistence=0.65)
    thr = 1.0 - min(0.5, target_area * 2.5) - np.random.uniform(0.06, 0.14)
    m = (base > thr).astype(np.float32)
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.8))
    m = np.asarray(m).astype(np.float32) / 255.0
    m = irregularize(m, jag=0.95, blur=1.3)
    m = area_limit(m, target_area)
    return m


def mask_silvering_edges(h, w, target_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
    yy, xx = np.ogrid[:h, :w]
    xx2 = np.broadcast_to(xx, (h, w)).astype(np.float32)
    yy2 = np.broadcast_to(yy, (h, w)).astype(np.float32)
    dist_edge = np.minimum(np.minimum(xx2, (w - 1 - xx2)), np.minimum(yy2, (h - 1 - yy2)))
    band = np.exp(-dist_edge / np.random.uniform(8, 18))
    noise = value_noise(h, w, base=14, octaves=3, persistence=0.6)
    m = band * (0.6 + 0.7 * noise)
    m = (m - m.mean()) > 0.0
    m = m.astype(np.float32)
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.4))
    m = np.asarray(m).astype(np.float32) / 255.0
    m = area_limit(m, target_area * 0.7)
    return m


def mask_light_leak(h, w, target_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
    corner = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((yy - corner[0]) ** 2 + (xx - corner[1]) ** 2)
    rr = np.random.uniform(0.3, 0.55) * max(h, w)
    grad = np.clip(1 - d / rr, 0, 1)
    noise = value_noise(h, w, base=16, octaves=3, persistence=0.6)
    m = grad * (0.55 + 0.65 * noise)
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.2))
    m = np.asarray(m).astype(np.float32) / 255.0
    m = area_limit(m, target_area * 0.6)
    return m

def combine_masks(masks, max_area):
    m = np.zeros_like(masks[0])
    for x in masks:
        if x is None:
            continue
        m = np.maximum(m, x)
    return area_limit(np.clip(m, 0, 1), max_area)


def mask_scribble(h, w, thickness=(6, 10), segments=(7, 12), jitter=0.10, blur=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    def chaikin(points, iters=3):
        pts = np.array(points, dtype=np.float32)
        for _ in range(iters):
            new = []
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new.extend([q, r])
            pts = np.array(new, dtype=np.float32)
        return pts

    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    n_seg = random.randint(*segments)

    # anchor points roughly spanning height, with jitter in both axes
    pts = []
    for i in range(n_seg):
        px = random.uniform(0.12, 0.88) * w
        py = i / (n_seg - 1 + 1e-6) * h
        px += random.uniform(-jitter, jitter) * w
        py += random.uniform(-jitter, jitter) * h
        pts.append((px, py))

    # smooth and add slight wiggle
    smooth = chaikin(pts, iters=3)
    wiggle = np.random.normal(scale=0.01 * min(h, w), size=smooth.shape)
    smooth += wiggle

    smooth_pts = [(float(x), float(y)) for x, y in smooth]
    thick = random.randint(*thickness)
    draw.line(smooth_pts, fill=255, width=thick, joint="curve")

    # optional secondary pass with thinner stroke for natural feel
    if random.random() < 0.5:
        k = random.uniform(0.6, 0.9)
        draw.line(smooth_pts, fill=255, width=int(max(1, thick * k)), joint="curve")

    m = np.asarray(canvas).astype(np.float32) / 255.0
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur))
    m = np.asarray(m).astype(np.float32) / 255.0
    return m


# -------- 缺陷应用 --------
def tone_shift(img_np, severity):
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    warm = np.stack([r * 1.03 + 0.03 * severity, g * 0.98, b * 0.95 - 0.02 * severity], axis=2)
    mean = warm.mean(axis=(0, 1), keepdims=True)
    warm = (warm - mean) * (1.0 - 0.1 * severity) + mean
    return np.clip(warm, 0, 1)


def apply_defects(clean_pil, mask, severity):
    clean = np.asarray(clean_pil).astype(np.float32) / 255.0
    h, w = clean.shape[:2]
    if mask is None or mask.max() <= 1e-6:
        return clean_pil

    base_aged = tone_shift(clean, min(severity * 0.7 + 0.1, 0.85))

    blur_rad = 0.6 + 2.5 * severity
    sharp_rad = max(0.0, 1.3 - 1.0 * severity)
    defect_local = Image.fromarray((base_aged * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blur_rad)
    )
    defect_local = np.asarray(defect_local).astype(np.float32) / 255.0

    delta = (np.random.uniform(-0.35, 0.35) * (0.7 + 0.6 * severity))
    defect_local = np.clip(defect_local + delta, 0, 1)

    grain = value_noise(h, w, base=8, octaves=4, persistence=0.55)
    grain = (grain - 0.5) * (0.3 + 0.4 * severity)
    defect_local = np.clip(defect_local + grain[..., None], 0, 1)

    opacity = 0.55 + 0.5 * severity
    alpha = np.clip(mask[..., None] * opacity, 0, 1)

    out = clean * (1.0 - alpha) + defect_local * alpha

    if sharp_rad > 0:
        sharp = Image.fromarray((out * 255).astype(np.uint8)).filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=int(90 + 80 * severity), threshold=3)
        )
        out = np.asarray(sharp).astype(np.float32) / 255.0

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# -------- 数据处理 --------
def process_dataset(
    src_dir: Path,
    out_dir: Path,
    size: int,
    severity: float,
    max_area: float,
    limit: int = -1,
    seed: int = 42,
    mode: str = "mixed",
):
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    img_paths = []
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            img_paths.append(str(p))
    img_paths = sorted(img_paths)
    if limit > 0:
        img_paths = img_paths[:limit]

    seed_all(seed)

    out_clean = out_dir / "clean"
    out_def = out_dir / "defects"
    out_mask = out_dir / "masks"
    out_clean.mkdir(parents=True, exist_ok=True)
    out_def.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(img_paths):
        clean = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        w, h = clean.size

        tgt_area = max_area * np.random.uniform(0.8, 1.2)

        if mode == "scribble":
            mask = mask_scribble(h, w, thickness=(6, 10), segments=(7, 12), jitter=0.10, blur=2.0, seed=seed * 31 + i)
            mask = area_limit(mask, tgt_area * 0.9)
        else:
            parts = [
                mask_mottle(h, w, tgt_area * 0.45, seed=seed * 11 + i),
                mask_dust_fiber(h, w, tgt_area * 0.25, seed=seed * 13 + i),
                mask_scratches(h, w, tgt_area * 0.25, seed=seed * 17 + i),
            ]
            if np.random.rand() < 0.7:
                parts.append(mask_silvering_edges(h, w, tgt_area * 0.3, seed=seed * 19 + i))
            if np.random.rand() < 0.6:
                parts.append(mask_light_leak(h, w, tgt_area * 0.25, seed=seed * 23 + i))
            # 大面积不规则缺损（随机形状+噪声扰动）
            if np.random.rand() < 0.6:
                blob = np.zeros((h, w), dtype=np.float32)
                rr = np.random.uniform(0.2, 0.5) * min(h, w)
                cx = np.random.randint(int(0.1 * w), int(0.9 * w))
                cy = np.random.randint(int(0.1 * h), int(0.9 * h))
                yy, xx = np.ogrid[:h, :w]
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                blob[dist < rr] = 1.0
                blob = irregularize(blob, jag=1.0, blur=2.5)
                blob = morph(blob, iters=np.random.randint(1, 3), mode=np.random.choice(["dilate", "erode"]))
                blob = area_limit(blob, tgt_area * np.random.uniform(0.4, 0.8))
                parts.append(blob)

            # 随机形态扰动
            parts = [
                morph(x, iters=np.random.randint(0, 2), mode=np.random.choice(["dilate", "erode"]))
                if x is not None
                else None
                for x in parts
            ]

            mask = combine_masks([x for x in parts if x is not None], max_area=max_area)

        if mode == "scribble":
            # 直接用纯白笔画覆盖
            m = (mask >= 0.3).astype(np.float32)[..., None]
            base = np.asarray(clean).astype(np.float32)
            defect_arr = base * (1.0 - m) + 255.0 * m
            defect = Image.fromarray(np.clip(defect_arr, 0, 255).astype(np.uint8))
        else:
            defect = apply_defects(clean, mask, severity=severity)

        base = f"def_{i:05d}.png"
        clean.save(out_clean / base)
        defect.save(out_def / base)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(out_mask / base)

        if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
            print(f"[{i+1}/{len(img_paths)}] saved {base}")


def main():
    parser = argparse.ArgumentParser(description="Build structural defects for FFHQ-like data")
    parser.add_argument("--src_dir", type=str, default="ffhq256/images", help="Source image directory")
    parser.add_argument("--out_dir", type=str, default="ffhq256_defect_strong", help="Output directory")
    parser.add_argument("--size", type=int, default=256, help="Resize images to this size")
    parser.add_argument("--severity", type=float, default=0.7, help="Defect strength 0~1")
    parser.add_argument("--max_area", type=float, default=0.12, help="Max defect area ratio 0~1")
    parser.add_argument("--limit", type=int, default=-1, help="Number of images to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "scribble"], help="mixed or scribble-only defects")
    args = parser.parse_args()

    process_dataset(
        Path(args.src_dir),
        Path(args.out_dir),
        size=args.size,
        severity=args.severity,
        max_area=args.max_area,
        limit=args.limit,
        seed=args.seed,
        mode=args.mode,
    )
    print("Done. Output ->", args.out_dir)


if __name__ == "__main__":
    main()
