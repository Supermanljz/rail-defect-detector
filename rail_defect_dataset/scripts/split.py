# -*- coding: utf-8 -*-
"""
训练/测试划分脚本
两种模式:
    1. 随机划分（默认 80/20，固定种子，对齐论文 Section 3.3）
        python scripts/split.py --src images/unlabeled_and_labeled_dir
    2. 按区段划分（防数据泄漏：测试集用训练没见过的区段，需配合 metadata csv）
        python scripts/split.py --by-section --metadata metadata.csv --test-sections "XX线K10"

工作流:
    先把已标注的图像放入 images/all/、标签放入 labels/all/（自行建立），
    运行本脚本后自动移动到 images/train|test 和 labels/train|test。
    划分完成后 images/all 应为空，可删除。
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def move_pair(img: Path, lbl: Path, split: str, root: Path, copied):
    dst_img = root / 'images' / split / img.name
    dst_lbl = root / 'labels' / split / lbl.name
    if copied:
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, dst_lbl)
    else:
        shutil.move(str(img), dst_img)
        shutil.move(str(lbl), dst_lbl)


def main():
    parser = argparse.ArgumentParser(description='训练/测试划分')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--src', default='all', help='来源子目录名（images/<src> 与 labels/<src>）')
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='固定随机种子，保证可复现')
    parser.add_argument('--by-section', action='store_true', help='按 metadata 中的区段列划分')
    parser.add_argument('--metadata', default='metadata.csv', help='元数据表路径（--by-section 时必填）')
    parser.add_argument('--test-sections', default='', help='划入测试集的区段关键字，分号分隔，如 "K10;K11"')
    parser.add_argument('--copy', action='store_true', help='复制而非移动（保留原图备份）')
    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root / 'images' / args.src
    lbl_dir = root / 'labels' / args.src
    if not img_dir.exists():
        print(f'[错误] 未找到 {img_dir}')
        print('请先把已标注图像放入该目录，标签放同名目录到 labels/ 下')
        sys.exit(1)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f'[错误] {img_dir} 中没有图像')
        sys.exit(1)

    # 检查图像-标签配对
    pairs = []
    missing = []
    for img in images:
        lbl = lbl_dir / f'{img.stem}.txt'
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing.append(img.name)
    if missing:
        for name in missing:
            print(f'[错误] 缺少标签: {name}')
        sys.exit(1)

    # 划分
    test_stems = set()
    if args.by_section:
        meta_path = Path(args.metadata)
        if not meta_path.exists():
            print(f'[错误] 未找到元数据表 {meta_path}')
            sys.exit(1)
        keywords = [k.strip() for k in args.test_sections.split(';') if k.strip()]
        if not keywords:
            print('[错误] --by-section 需要用 --test-sections 指定测试区段关键字')
            sys.exit(1)
        with open(meta_path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                section = (row.get('线路/区段') or '').strip()
                fname = (row.get('文件名') or '').strip()
                if not fname:
                    continue
                if any(k in section for k in keywords):
                    test_stems.add(Path(fname).stem)
        print(f'元数据中命中测试区段 {keywords} 的图像: {len(test_stems)} 张')
    else:
        random.seed(args.seed)
        stems = [img.stem for img, _ in pairs]
        n_test = round(len(stems) * args.test_ratio)
        test_stems = set(random.sample(stems, n_test))

    # 移动/复制
    n_train = n_test_moved = 0
    for img, lbl in pairs:
        split = 'test' if img.stem in test_stems else 'train'
        move_pair(img, lbl, split, root, args.copy)
        if split == 'test':
            n_test_moved += 1
        else:
            n_train += 1

    print(f'完成: train {n_train} 张, test {n_test_moved} 张'
          f'（{"复制" if args.copy else "移动"}模式, seed={args.seed}）')
    print('\n提示: 划分后请运行 python scripts/validate.py 复核格式')


if __name__ == '__main__':
    main()
