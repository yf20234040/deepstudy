import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        root: 包含 'PNGImages' 和 'PedMasks' 的根目录（PennFudanPed 数据集标准结构）
        transforms: torchvision.transforms 或自定义 transform（对 PIL image 和 target）
        """
        self.root = root
        self.transforms = transforms
        # images 和 masks 目录名按官方结构
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # 每个实例在 mask 中用不同的灰度值标识，0 为背景
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # 去掉背景

        # 分离每个实例的二进制 mask
        masks = mask == obj_ids[:, None, None]

        # 计算包围盒 (xmin, ymin, xmax, ymax)
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                # 空实例保护（极少见）
                xmin = ymin = xmax = ymax = 0
            else:
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转为 torch tensors，注意 Mask R-CNN 要求 boxes 为 float (x1,y1,x2,y2)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 这里只有一个类别：前景（person）
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是 crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target