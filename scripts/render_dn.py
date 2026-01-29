import imageio
import numpy as np
import sys

fp = sys.argv[1]
video = imageio.get_reader(fp,  'ffmpeg')
frames = np.array([f for f in video])
H, W = frames.shape[1:3]

rgb = frames[:, :, :W//3]           # RGB
depth = frames[:, :, W//3:W//3*2]   # Depth  
normal = frames[:, :, W//3*2:]      # Normal

# 保存为单独视频
imageio.mimwrite(f"rgb-{fp}", rgb, fps=8)
imageio.mimwrite(f"depth-{fp}", depth, fps=8)
imageio.mimwrite(f"normal-{fp}", normal, fps=8)

