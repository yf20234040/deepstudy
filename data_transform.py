import torch
from torchvision import transforms as T  # 改为从torchvision导入transforms
from torch.utils.data import DataLoader, Subset
from torchvision import utils
from pennfudan_dataset import PennFudanDataset

# 定义数据转换函数（训练集添加水平翻转扩充）
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())  # 转为张量并归一化
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # 50%概率水平翻转
    return T.Compose(transforms)

# 加载数据集并分割训练集/测试集
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
indices = torch.randperm(len(dataset)).tolist()  # 随机打乱索引
dataset = Subset(dataset, indices[:-50])  # 训练集（除最后50张）
dataset_test = Subset(dataset_test, indices[-50:])  # 测试集（最后50张）

# 定义数据加载器（collate_fn处理批次中不同数量的目标）
data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)
data_loader_test = DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
)