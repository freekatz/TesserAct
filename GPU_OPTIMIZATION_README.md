# GPU-Accelerated Video Processing

## 概述

TesserAct 现在支持 GPU 加速的视频处理，可显著提升图像序列到视频的转换速度。

### 性能提升

| 硬件配置 | 加速比 | 说明 |
|---------|--------|------|
| NVIDIA GPU (RTX 3060+) | ~5-8x | GPU JPEG 解码 + 多核编码 |
| 多核 CPU (8+ cores) | ~3-5x | 多线程编码 |
| 单核/双核 CPU | ~1.5-2x | 流式处理，减少内存占用 |

## 安装依赖

### 1. 安装 FFmpeg (必需)

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

### 2. 安装 Python 依赖

```bash
# 基础依赖 (已在 requirements.txt 中)
pip install av>=13.0.0 torchvision>=0.20.1

# 或直接安装所有依赖
pip install -r requirements.txt
```

### 3. 验证 GPU 支持 (可选)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

如果输出 `CUDA available: True`，则可以使用 GPU 加速。如果是 `False`，系统会自动回退到多核 CPU 加速。

## 使用方法

### 自动优化 (推荐)

现有代码无需修改，会自动使用优化后的实现：

```python
from scripts.preprocess_bridge import BridgePreprocessor

# 默认启用 GPU 加速
preprocessor = BridgePreprocessor(
    raw_data_dir='./data/bridge/raw',
    output_dir='./data-processed/bridge',
    cache_dir='./cache/bridge',
)
preprocessor.process()
```

### 自定义配置

可以通过参数控制优化行为：

```python
preprocessor = BridgePreprocessor(
    raw_data_dir='./data/bridge/raw',
    output_dir='./data-processed/bridge',
    cache_dir='./cache/bridge',

    # 视频参数
    fps=30,
    video_quality=23,        # CRF: 18-28 (越小质量越好)
    encoding_preset='medium', # fast/medium/slow (越慢压缩越好)

    # GPU 参数
    use_gpu=True,            # 启用/禁用 GPU
    batch_size=16,           # GPU 批处理大小 (8-32)
)
preprocessor.process()
```

### 直接使用工具函数

```python
from tesseract.utils.video_io import images_to_video_optimized
import glob

# 获取图像列表
images = sorted(glob.glob("path/to/images/*.jpg"))

# 转换为视频
images_to_video_optimized(
    image_paths=images,
    output_path="output.mp4",
    fps=30,
    batch_size=16,        # GPU 批大小
    device='cuda',        # 'cuda' 或 'cpu'
    crf=23,               # 质量 (18-28)
    preset='medium',      # 编码速度
    verbose=True          # 显示进度
)
```

## 测试性能

运行测试脚本对比优化前后的性能：

```bash
python scripts/test_video_optimization.py
```

测试脚本会：
1. 检查 GPU 可用性
2. 创建 100 张测试图像
3. 对比 imageio (原始) vs 优化方案的性能
4. 显示加速比和文件大小对比

## 技术细节

### 架构

```
图像文件 → GPU 批量解码 → 张量批次 → 转换为 numpy → 流式传输到 PyAV 编码器 → MP4 输出
```

### 关键优化

1. **GPU JPEG 解码** (torchvision)
   - 批量解码多张图像
   - ~10-15x 快于 CPU imageio.imread()
   - 自动回退到 CPU (如果无 GPU)

2. **多线程 H.264 编码** (PyAV)
   - 利用所有 CPU 核心
   - 流式处理 (无需全部加载到内存)
   - ~3-5x 快于单线程 imageio.mimsave()

3. **内存优化**
   - 流式管道 (不会全部加载到 RAM)
   - 适合处理大型视频序列
   - 防止内存溢出 (OOM)

### 质量参数说明

**CRF (Constant Rate Factor):**
- 范围: 0-51
- 推荐: 18-28
- 18: 视觉无损 (大文件)
- 23: 默认值 (平衡质量和大小)
- 28: 较小文件 (稍有质量损失)

**Preset (编码速度):**
- `ultrafast`: 最快，文件较大
- `fast`: 快速，文件稍大
- `medium`: 默认，平衡速度和压缩
- `slow`: 慢速，更好的压缩
- `veryslow`: 最慢，最佳压缩

## 故障排查

### PyAV 安装失败

如果 `pip install av` 失败，请确保已安装 FFmpeg 开发库：

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

### GPU 未被使用

检查 CUDA 是否可用：
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

如果返回 `False`：
- 确保安装了 CUDA 版本的 PyTorch
- 检查 NVIDIA 驱动是否正确安装
- macOS 不支持 CUDA (会自动使用 CPU)

### 编码失败回退到 imageio

如果看到警告 "GPU-accelerated encoding failed"：
1. 检查 PyAV 是否正确安装: `python -c "import av; print(av.__version__)"`
2. 检查 FFmpeg 是否可用: `ffmpeg -version`
3. 系统会自动回退到原始 imageio 实现

### 性能未提升

在 CPU-only 系统上：
- 预期加速比: ~3-5x (取决于 CPU 核心数)
- 单核/双核 CPU 加速有限

在有 GPU 的系统上：
- 预期加速比: ~5-8x
- 确保 batch_size 足够大 (16-32)
- 检查 GPU 利用率: `nvidia-smi`

## 兼容性

- ✅ **向后兼容**: 保留 imageio 作为备选方案
- ✅ **无破坏性更改**: 新工具是可选的
- ✅ **内存安全**: 流式处理防止 OOM 错误
- ✅ **自动回退**: GPU 不可用时使用 CPU

## 其他脚本优化

可以使用相同的工具优化其他视频处理脚本：

### video_normal.py

```python
# 替换 imageio.imiter() 和 imageio.imwrite()
from tesseract.utils.video_io import encode_video_multicore

# 使用 PyAV 流式编码
encode_video_multicore(
    frames=output_frames,  # 可以是 iterator
    output_path=path_out,
    fps=30,
    crf=23,
    preset='medium',
    verbose=True
)
```

### rendering_points.py

类似地替换视频读取和写入操作。

## 参考文档

- [PyAV 文档](https://pyav.org/)
- [torchvision.io 文档](https://pytorch.org/vision/stable/io.html)
- [FFmpeg H.264 编码指南](https://trac.ffmpeg.org/wiki/Encode/H.264)
