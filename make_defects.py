import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import json
from tqdm import tqdm

def add_random_mask(img, num_shapes=3, min_size=0.05, max_size=0.2):
    """
    在图片上添加随机不规则白色遮挡。
    :param img: PIL Image
    :param num_shapes: 遮挡块数量
    :param min_size: 遮挡块相对宽度比例下界
    :param max_size: 遮挡块相对宽度比例上界
    """
    w, h = img.size
    # 在掩膜上画多边形，允许随机不透明度，再用掩膜叠加纯白
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(num_shapes):
        # 随机多边形的顶点数量
        num_points = random.randint(4, 8)
        # 随机生成多边形顶点
        poly = []
        scale = random.uniform(min_size, max_size)
        pw = int(w * scale)
        ph = int(h * scale)
        # 随机选取多边形中心
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        for _ in range(num_points):
            x = cx + random.randint(-pw, pw)
            y = cy + random.randint(-ph, ph)
            poly.append((x, y))
        alpha = random.uniform(0.5, 1.0)
        draw.polygon(poly, fill=int(255 * alpha))
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    img = Image.composite(white_bg, img, mask)
    return img

def add_random_strokes(img, num_strokes=3, min_len_ratio=0.1, max_len_ratio=0.3, min_width=3, max_width=15):
    """
    添加随机白色柔和笔触/划痕。
    :param img: PIL Image
    :param num_strokes: 笔触数量
    :param min_len_ratio: 笔触长度相对短边的下界
    :param max_len_ratio: 笔触长度相对短边的上界
    :param min_width: 笔触最小宽度（像素）
    :param max_width: 笔触最大宽度（像素）
    """
    w, h = img.size
    ref = min(w, h)

    # 在灰度掩膜上绘制，避免黑边，再用掩膜将纯白笔触叠加
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    def smooth_points(pts, iters=3):
        for _ in range(iters):
            new_pts = []
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])
            pts = [pts[0]] + new_pts + [pts[-1]]
        return pts

    for _ in range(num_strokes):
        length = random.uniform(min_len_ratio, max_len_ratio) * ref
        num_points = random.randint(8, 16)
        x, y = random.uniform(0, w), random.uniform(0, h)
        points = [(x, y)]
        step = length / num_points
        for _ in range(num_points - 1):
            dx = step * random.uniform(-1.5, 1.5)
            dy = step * random.uniform(-1.5, 1.5)
            x = min(max(x + dx, 0), w)
            y = min(max(y + dy, 0), h)
            points.append((x, y))
        smooth_pts = smooth_points(points, iters=2)
        width = random.randint(min_width, max_width)
        alpha = random.uniform(0.6, 1.0)  # 笔触不透明度
        draw.line(smooth_pts, fill=int(255 * alpha), width=width)

    # 用掩膜把纯白笔触叠加到原图
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    img = Image.composite(white_bg, img, mask)
    return img

def main():
    parser = argparse.ArgumentParser(description="Make defect dataset by adding random white masks.")
    parser.add_argument("--input_dir", type=str, default="ffhq_256/images", help="Source images directory")
    parser.add_argument("--output_dir", type=str, default="ffhq_256_defect/images", help="Output directory for masked images")
    parser.add_argument("--json_out", type=str, default="ffhq_256_defect/ffhq.json", help="Output json file")
    parser.add_argument("--num_shapes", type=int, default=7, help="Number of random shapes per image")
    parser.add_argument("--min_size", type=float, default=0.05, help="Min relative size of mask")
    parser.add_argument("--max_size", type=float, default=0.2, help="Max relative size of mask")
    parser.add_argument("--num_strokes", type=int, default=4, help="Number of random strokes per image")
    parser.add_argument("--stroke_min_len", type=float, default=3.0, help="Min relative length of stroke (vs short side)")
    parser.add_argument("--stroke_max_len", type=float, default=5.0, help="Max relative length of stroke (vs short side)")
    parser.add_argument("--stroke_min_width", type=int, default=3, help="Min stroke width in pixels")
    parser.add_argument("--stroke_max_width", type=int, default=5, help="Max stroke width in pixels")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    image_files = sorted([p for p in input_dir.glob("*.png")])
    for p in tqdm(image_files, desc="Processing"):
        img = Image.open(p).convert("RGB")
        img_masked = add_random_mask(img, num_shapes=args.num_shapes, min_size=args.min_size, max_size=args.max_size)
        img_masked = add_random_strokes(
            img_masked,
            num_strokes=args.num_strokes,
            min_len_ratio=args.stroke_min_len,
            max_len_ratio=args.stroke_max_len,
            min_width=args.stroke_min_width,
            max_width=args.stroke_max_width,
        )
        out_path = output_dir / p.name
        img_masked.save(out_path)
        # 生成 json 记录：image_file 是缺陷图，原图路径可按需要记录
        items.append({"image_file": p.name, "text": ""})

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"Saved masked images to {output_dir}, json to {json_out}, total {len(items)}")

if __name__ == "__main__":
    main()
