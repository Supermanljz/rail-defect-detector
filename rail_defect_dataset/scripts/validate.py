# -*- coding: utf-8 -*-
"""
数据集格式校验脚本
用法:
    python scripts/validate.py                # 校验 train + test
    python scripts/validate.py --split train  # 只校验 train
检查项:
    1. 图像与标签文件配对（缺失/多余）
    2. 标签文件为空（可能是负样本，警告）
    3. 非法类别 id（超出 classes.txt 范围）
    4. 坐标越界（不在 0~1）或宽高非法（<=0）
    5. 每行字段数不是 5
"""
import argparse
import sys
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_image(labels_dir: Path, stem: str):
    """根据标签文件名（不含扩展名）查找对应图像"""
    for ext in IMG_EXTS:
        p = labels_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def validate_split(root: Path, split: str, classes):
    img_dir = root / 'images' / split
    lbl_dir = root / 'labels' / split
    if not img_dir.exists() or not lbl_dir.exists():
        print(f'[跳过] {split}: 目录不存在')
        return 0

    images = {p.stem: p for p in img_dir.iterdir()
              if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in lbl_dir.glob('*.txt')}
    errors, warnings = [], []

    # 1. 配对检查
    for stem in images:
        if stem not in labels:
            errors.append(f'图像缺少标签: {images[stem].name}')
    for stem in labels:
        if stem not in images:
            errors.append(f'标签缺少图像: {labels[stem].name}')

    # 2-5. 标签内容检查（utf-8-sig 兼容 Windows 工具写入的 BOM 头）
    for stem, lbl in sorted(labels.items()):
        lines = lbl.read_text(encoding='utf-8-sig').strip().splitlines()
        if not lines:
            warnings.append(f'空标签（若为负样本可忽略）: {lbl.name}')
            continue
        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f'{lbl.name}:{i} 字段数={len(parts)}，应为 5')
                continue
            try:
                cid, cx, cy, w, h = (float(x) for x in parts)
            except ValueError:
                errors.append(f'{lbl.name}:{i} 存在非数值字段')
                continue
            if not cid.is_integer() or not (0 <= int(cid) < len(classes)):
                errors.append(f'{lbl.name}:{i} 非法类别 id: {cid}（应为 0~{len(classes)-1}）')
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                errors.append(f'{lbl.name}:{i} 中心坐标越界: cx={cx}, cy={cy}')
            if not (0 < w <= 1 and 0 < h <= 1):
                errors.append(f'{lbl.name}:{i} 宽高非法: w={w}, h={h}')

    print(f'\n===== {split} =====')
    print(f'图像 {len(images)} 张，标签 {len(labels)} 个')
    for msg in warnings:
        print(f'[警告] {msg}')
    for msg in errors:
        print(f'[错误] {msg}')
    if not errors and not warnings:
        print('全部通过')
    return len(errors)


def main():
    parser = argparse.ArgumentParser(description='YOLO 数据集格式校验')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent),
                        help='数据集根目录（默认为脚本上一级）')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    args = parser.parse_args()

    root = Path(args.root)
    classes_file = root / 'classes.txt'
    if not classes_file.exists():
        print(f'[错误] 未找到 {classes_file}')
        sys.exit(1)
    classes = classes_file.read_text(encoding='utf-8').strip().splitlines()
    print(f'类别（{len(classes)} 类）: {", ".join(classes)}')

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    total_errors = sum(validate_split(root, s, classes) for s in splits)
    if total_errors:
        print(f'\n共 {total_errors} 个错误，请修复后再训练')
        sys.exit(1)
    print('\n校验完成，无错误')


if __name__ == '__main__':
    main()
