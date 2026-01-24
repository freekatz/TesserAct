#!/bin/bash
# 验证 GPU 优化更改的脚本

echo "========================================"
echo "验证 GPU 优化更改"
echo "========================================"

echo ""
echo "1. 检查文件是否存在..."
files=(
    "tesseract/utils/__init__.py"
    "tesseract/utils/video_io.py"
    "scripts/test_video_optimization.py"
    "GPU_OPTIMIZATION_README.md"
    "CHANGES_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
    fi
done

echo ""
echo "2. 检查 requirements.txt 更新..."
if grep -q "av>=13.0.0" requirements.txt; then
    echo "  ✓ PyAV 依赖已添加"
else
    echo "  ✗ PyAV 依赖缺失"
fi

echo ""
echo "3. 检查 preprocess_bridge.py 更新..."
if grep -q "images_to_video_optimized" scripts/preprocess_bridge.py; then
    echo "  ✓ 导入了 images_to_video_optimized"
else
    echo "  ✗ 未导入 images_to_video_optimized"
fi

if grep -q "batch_size" scripts/preprocess_bridge.py; then
    echo "  ✓ 添加了 batch_size 参数"
else
    echo "  ✗ 缺少 batch_size 参数"
fi

if grep -q "Falling back to imageio" scripts/preprocess_bridge.py; then
    echo "  ✓ 包含回退逻辑"
else
    echo "  ✗ 缺少回退逻辑"
fi

echo ""
echo "4. 检查 Python 语法..."
python -m py_compile scripts/preprocess_bridge.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ preprocess_bridge.py 语法正确"
else
    echo "  ✗ preprocess_bridge.py 语法错误"
fi

python -m py_compile tesseract/utils/video_io.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ video_io.py 语法正确"
else
    echo "  ✗ video_io.py 语法错误"
fi

echo ""
echo "5. 检查依赖 (仅检查已安装的)..."

python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "  ⚠ PyTorch 未安装"
python -c "import torchvision; print('  ✓ torchvision:', torchvision.__version__)" 2>/dev/null || echo "  ⚠ torchvision 未安装"
python -c "import av; print('  ✓ PyAV:', av.__version__)" 2>/dev/null || echo "  ⚠ PyAV 未安装 (需要在开发机安装)"

echo ""
echo "6. 检查 GPU 可用性..."
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())" 2>/dev/null

echo ""
echo "========================================"
echo "验证完成！"
echo ""
echo "下一步 (在开发机上):"
echo "  1. pip install -r requirements.txt"
echo "  2. python scripts/test_video_optimization.py"
echo "  3. python scripts/preprocess_bridge.py"
echo "========================================"
