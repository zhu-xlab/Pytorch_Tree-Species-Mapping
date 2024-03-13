import numpy as np
import torch
import rasterio
import os
from torchvision import transforms
import argparse

from utils.config_utils import get_config_from_json
from models.Forestformer import CustomTransformerModel
from data_loader import MyDataLoader

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Run the experiment with custom parameters')
parser.add_argument('--gpu', required=True, help='Index of gpu used')
parser.add_argument('--tifpath', required=True, help='Path to the GeoTiff file')
# parser.add_argument('--exp_name', required=True, help='Experiment name')
# parser.add_argument('--rpath', required=True, help='Path to the inference GeoTiff file')

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数
gpu = args.gpu
tifpath = args.tifpath
# exp_name = args.exp_name
# rpath = args.rpath


# torch.set_num_threads(18)

# Set the GPU device index to use
opt = {"gpu_ids": [gpu]}
# opt['gpu_ids'] = [2, 3]
gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
# gpu_list = str(opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# 加载模型NI
cfg = 'forestformer'
config = get_config_from_json('./configs/'+cfg+'.json')
exp_name = config['exp_name']
checkpoint_path = "./experiments/"+exp_name+"/checkpoints/best_model.pth"

device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
print(f"Predicting on {device}")

#***********************************************************************************************************************
model = CustomTransformerModel(config=config)

new_state_dict = model.state_dict()
checkpoint = torch.load(checkpoint_path)
for name, param in checkpoint.named_parameters():
    if name in new_state_dict:
        new_state_dict[name] = param
model.load_state_dict(new_state_dict)

model = model.to(device)

#***********************************************************************************************************************
# 打开遥感影像文件
# tifpath = './data/data_map/tifformap/DE22/'
tiffiles = [f for f in os.listdir(tifpath) if f.endswith('.tif')]
print(tiffiles)
rpath = './data_map/rformap/1109/'

dl = MyDataLoader(config=config)
mean, std = dl.get_mean_std()

for tif in tiffiles:
    tif_path = os.path.join(tifpath, tif)
    rfile = 'r_' + tif
    r_path = os.path.join(rpath, rfile)
    if os.path.exists(r_path):  # 调用exist函数检查是否存在
        print(str(r_path) + ' exists')
        continue
    with rasterio.open(tif_path) as src:
        # 计算图像宽度和高度
        height, width = src.shape

        # 定义输出图像文件
        profile = src.profile
        profile.update(dtype=rasterio.int8, count=1, nodata=11)

        with rasterio.open(r_path, 'w', **profile) as dst:
            # 计算需要分成多少个 patch
            patch_size = (5, 5, 120)
            num_patches = (height // patch_size[0], width // patch_size[1])

            # 循环遍历每个 patch
            for i in range(num_patches[0]):
                for j in range(num_patches[1]):
                    # 读取当前 patch
                    patch = src.read(
                        window=rasterio.windows.Window(j * patch_size[1], i * patch_size[0], patch_size[1], patch_size[0]))

                    # 检查 patch 中是否存在缺失值
                    # if np.isnan(patch).any():
                    #     continue

                    if np.sum(np.isnan(patch[:, :, 0])) > 12:
                        continue

                    # 将 patch 转换为 pytorch tensor
                    patch = torch.from_numpy(patch).float().unsqueeze(0)  #(1, 120, 5, 5)
                    patch = transforms.Normalize(mean, std)(patch)
                    if config.model == 'convlstm':
                        patch = torch.reshape(patch, (1, config.num_length, config.input_channels, patch_size[0], patch_size[1]))

                    # 执行预测
                    with torch.no_grad():
                        patch = patch.to(device)
                        output = model(patch)

                        _, predicted = torch.max(output, 1)
                        prediction = predicted.cpu().numpy()[0]

                        pred_patch = np.full((5, 5), prediction)

                    # 将预测结果写入输出图像文件
                    dst.write(pred_patch.astype(rasterio.int8), 1,
                              window=rasterio.windows.Window(j * patch_size[1], i * patch_size[0], patch_size[1],
                                                             patch_size[0]))
