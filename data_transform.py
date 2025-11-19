import torch
# 从自定义transforms导入正确的类
from transforms import Compose, PILToTensor, ToDtype, RandomHorizontalFlip
from torch.utils.data import DataLoader, Subset
from pennfudan_dataset import PennFudanDataset

# 定义适用于目标检测的collate函数
def collate_fn(batch):
    return tuple(zip(*batch))

# 定义数据转换函数（使用自定义转换类）
def get_transform(train):
    transforms = []
    # 1. 将PIL图像转为张量（对应原ToTensor的第一步）
    transforms.append(PILToTensor())
    # 2. 转换数据类型并归一化（对应原ToTensor的归一化步骤）
    transforms.append(ToDtype(torch.float32, scale=True))
    # 训练集添加水平翻转
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)  # 使用自定义Compose处理双参数

# 加载数据集并分割训练集/测试集
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
indices = torch.randperm(len(dataset)).tolist()
dataset = Subset(dataset, indices[:-50])
dataset_test = Subset(dataset_test, indices[-50:])

# 定义数据加载器
data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
)
data_loader_test = DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
)