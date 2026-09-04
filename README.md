# rail-defect-detector

钢轨表面缺陷检测项目（复现 Rail-5k 论文基线：arXiv:2106.14366）。
跨设备开发：Windows（RTX 4060，CUDA）/ macOS（M5 Pro 64GB，MPS）。

## 环境管理：pip + venv（本仓库不用 uv）

> ⚠️ **不要在本目录执行 `uv sync`**：它会按 pyproject.toml 的空依赖列表清空重建 .venv，环境报废，且 torch 需重新下载 2.5GB。

### Windows（NVIDIA GPU）

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"  # 验证应输出 True
```

### macOS（Apple Silicon，MPS 加速）

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"  # 验证应输出 True
```

### 运行测试

```powershell
.venv\Scripts\python.exe -m pytest   # Windows
.venv/bin/python -m pytest           # macOS
```

## 项目结构

```
rail-defect-detector/
├── requirements.txt        # 依赖真源（跨平台，torch 无 +cu126 后缀）
├── .python-version         # Python 3.13
├── pyproject.toml          # 打包元数据（暂闲置；依赖由 pip 管理）
├── rail_defect_dataset/    # 数据集 + 工具脚本（数据不进 Git）
│   ├── scripts/            # split / validate / stats
│   ├── classes.txt         # 11 类，id 已冻结不可改
│   └── data.yaml           # YOLO 训练配置
├── src/rail_defect_detector/
└── tests/
```

## 数据同步

图像/标注体积大，**不进 Git**。两台机器通过网盘/移动硬盘同步以下目录，
目录结构两边保持一致（脚本均为相对路径，换机器零修改）：

- `rail_defect_dataset/images/`
- `rail_defect_dataset/labels/`
- `rail_defect_dataset/metadata*.csv`

## 双机分工

- **主力训练：macOS（M5 Pro 64GB）** —— 统一内存大，1280 高分辨率训练与 SAHI 大图切图无压力
- **备用/基准：Windows（RTX 4060 8GB）** —— CUDA 生态兼容性最佳；MPS 个别算子会回退 CPU，首次训练前建议两边各跑一版小规模 benchmark 对比后再定主力
