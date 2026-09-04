# -*- coding: utf-8 -*-
"""
数据集统计可视化脚本（对齐论文 Rail-5k Table 2 / Figure 5-7）
用法:
    python scripts/stats.py                # 统计 train + test
    python scripts/stats.py --split train  # 只统计 train
输出:
    终端打印类别统计表（框数/图像数/大中小目标数）
    reports/ 下生成 4 张图:
        class_distribution.png  类别分布（框数 vs 图像数）
        box_sizes.png           框尺寸分布（按图像面积占比分大/中/小）
        wh_ratio.png            宽高比分布
        center_heatmap.png      目标中心点热力图
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无界面环境
import matplotlib.pyplot as plt

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 按框面积占整图面积的比例划分（自采数据分辨率可能不同，用相对面积更稳）
# 可按需调整；论文用的是 COCO 式绝对像素阈值（32^2 / 96^2）
SMALL_MAX = 0.001   # area_ratio < 0.1% 为小目标
MEDIUM_MAX = 0.01   # 0.1% ~ 1% 为中目标，> 1% 为大目标


def load_boxes(lbl_dir: Path):
    """返回 [(class_id, cx, cy, w, h), ...]"""
    boxes = []
    for lbl in sorted(lbl_dir.glob('*.txt')):
        for line in lbl.read_text(encoding='utf-8-sig').strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid, cx, cy, w, h = (float(x) for x in parts)
            except ValueError:
                continue
            boxes.append((int(cid), cx, cy, w, h))
    return boxes


def collect(root: Path, split: str):
    lbl_dir = root / 'labels' / split
    img_dir = root / 'images' / split
    if not lbl_dir.exists():
        return None
    boxes = load_boxes(lbl_dir)
    # 统计每类出现的图像数
    img_set = {}
    for lbl in lbl_dir.glob('*.txt'):
        cids = set()
        for line in lbl.read_text(encoding='utf-8-sig').strip().splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0].isdigit():
                cids.add(int(parts[0]))
        for cid in cids:
            img_set.setdefault(cid, set()).add(lbl.stem)
    n_images = len([p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else 0
    return boxes, img_set, n_images


def main():
    parser = argparse.ArgumentParser(description='数据集统计可视化')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    args = parser.parse_args()

    root = Path(args.root)
    classes = (root / 'classes.txt').read_text(encoding='utf-8').strip().splitlines()
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    all_boxes, all_img_set = [], {}
    for split in splits:
        result = collect(root, split)
        if result is None:
            print(f'[跳过] {split}: labels/{split} 不存在')
            continue
        boxes, img_set, n_images = result
        all_boxes += boxes
        for cid, stems in img_set.items():
            all_img_set.setdefault(cid, set()).update(stems)
        print(f'{split}: 图像 {n_images} 张，标注框 {len(boxes)} 个')

    if not all_boxes:
        print('没有标注数据，无法生成统计图')
        return

    n_cls = len(classes)
    box_cnt = Counter(b[0] for b in all_boxes)
    img_cnt = {cid: len(all_img_set.get(cid, set())) for cid in range(n_cls)}
    size_cnt = {cid: [0, 0, 0] for cid in range(n_cls)}  # [大, 中, 小]

    for cid, cx, cy, w, h in all_boxes:
        ratio = w * h
        idx = 0 if ratio > MEDIUM_MAX else (1 if ratio > SMALL_MAX else 2)
        size_cnt[cid][idx] += 1

    # 终端统计表（对齐论文 Table 2）
    print(f'\n{"类别":<20}{"框数":>8}{"图像数":>8}{"大":>8}{"中":>8}{"小":>8}')
    for cid in range(n_cls):
        name = classes[cid] if cid < len(classes) else f'未知id{cid}'
        l, m, s = size_cnt[cid]
        print(f'{name:<20}{box_cnt.get(cid, 0):>8}{img_cnt.get(cid, 0):>8}{l:>8}{m:>8}{s:>8}')

    # 图 1: 类别分布
    names = [classes[c] for c in range(n_cls)]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(n_cls)
    ax.bar([i - 0.2 for i in x], [box_cnt.get(c, 0) for c in range(n_cls)],
           width=0.4, label='# boxes')
    ax.bar([i + 0.2 for i in x], [img_cnt.get(c, 0) for c in range(n_cls)],
           width=0.4, label='# images')
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Class distribution (log scale)')
    fig.tight_layout()
    fig.savefig(reports / 'class_distribution.png', dpi=150)

    # 图 2: 框尺寸分布
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(list(x), [size_cnt[c][0] for c in range(n_cls)], label='large', width=0.6)
    ax.bar(list(x), [size_cnt[c][1] for c in range(n_cls)], bottom=[size_cnt[c][0] for c in range(n_cls)],
           label='medium', width=0.6)
    ax.bar(list(x), [size_cnt[c][2] for c in range(n_cls)],
           bottom=[size_cnt[c][0] + size_cnt[c][1] for c in range(n_cls)],
           label='small', width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Box size distribution (relative to image area)')
    fig.tight_layout()
    fig.savefig(reports / 'box_sizes.png', dpi=150)

    # 图 3: 宽高比分布
    ratios = [w / h for _, _, _, w, h in all_boxes if h > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=50, range=(0, 10))
    ax.set_xlabel('width / height')
    ax.set_ylabel('count')
    ax.set_title('Width-height ratio distribution')
    fig.tight_layout()
    fig.savefig(reports / 'wh_ratio.png', dpi=150)

    # 图 4: 中心点热力图
    import numpy as np
    heat = np.zeros((100, 100))
    for _, cx, cy, _, _ in all_boxes:
        ix, iy = min(int(cx * 100), 99), min(int(cy * 100), 99)
        heat[iy, ix] += 1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heat, cmap='hot', origin='lower')
    ax.set_title('Object center heatmap')
    fig.tight_layout()
    fig.savefig(reports / 'center_heatmap.png', dpi=150)

    print(f'\n统计图已保存到 {reports}')


if __name__ == '__main__':
    main()
