# train_and_test.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from pennfudan_dataset import PennFudanDataset
from MaskRCNN import get_instance_segmentation_model


# ======================
#   GPU 设置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================
#   可被 pickle 的 Transform 类
# ======================
class ComposeTransforms:
    def __init__(self, train=False):
        self.train = train

    def __call__(self, image, target):
        # PIL -> Tensor
        image = F.to_tensor(image)

        # 训练增强：随机水平翻转
        if self.train and torch.rand(1).item() > 0.5:
            # flip image
            image = image.flip(-1)

            # flip mask
            target["masks"] = target["masks"].flip(-1)

            # flip boxes
            w = image.shape[2]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        return image, target


# ======================
#   collate_fn
# ======================
def collate_fn(batch):
    return tuple(zip(*batch))


# ======================
#   训练一个 epoch
# ======================
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()

    for i, (images, targets) in enumerate(data_loader):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch {epoch} | Iter {i}/{len(data_loader)} | Loss: {losses.item():.4f}")


# ======================
#   简单验证（不算 mAP）
# ======================
def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            if i == 0:
                print("Validation sample output keys:", outputs[0].keys())

            if i >= 2:
                break


# ======================
#   入口
# ======================
def main():

    dataset_root = "./PennFudanPed"  # 修改成你的路径

    # 使用新的 Transform 类（不会再报 pickle 错）
    dataset = PennFudanDataset(dataset_root, transforms=ComposeTransforms(train=True))
    dataset_test = PennFudanDataset(dataset_root, transforms=ComposeTransforms(train=False))

    # 拆分训练集 / 验证集
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(indices))

    dataset_train = torch.utils.data.Subset(dataset, indices[:split])
    dataset_val = torch.utils.data.Subset(dataset_test, indices[split:])

    # Windows 必须 num_workers = 0
    data_loader = DataLoader(dataset_train, batch_size=2, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)

    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)

    # Model
    num_classes = 2  # person + background
    model = get_instance_segmentation_model(num_classes, pretrained_backbone=True)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Train
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        evaluate(model, data_loader_val, device)

        # Save checkpoint
        ckpt = f"maskrcnn_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
