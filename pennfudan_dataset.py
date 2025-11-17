import os
import numpy as np
import torch
from PIL import Image

# 数据集定义
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 排序图像和掩码文件，确保一一对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")  # 图像转为RGB格式
        mask = Image.open(mask_path)
        mask = np.array(mask)  # 掩码转为numpy数组，不同颜色代表不同实例

        # 提取实例ID（排除背景0）
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # 生成每个实例的二进制掩码
        masks = mask == obj_ids[:, None, None]

        # 计算每个实例的边界框（xmin, ymin, xmax, ymax）
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转换为torch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 仅1类（行人）+背景
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])  # 图像唯一ID
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 边界框面积
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # 无人群实例

        # 构建目标字典（符合torchvision检测模型输入要求）
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 应用数据转换（如归一化、水平翻转）
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)  # 返回数据集大小