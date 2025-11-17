import torch
from MaskRCNN import model, device  # 导入模型和设备
from data_transform import dataset_test  # 导入测试集
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F



def show_prediction(img, target, pred):
    # 图像反归一化（从张量转为PIL图像）
    img = F.to_pil_image(img)
    plt.figure(figsize=(12, 6))

    # 绘制原始图像+真实边界框
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    boxes = target["boxes"].cpu().numpy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'g-', linewidth=2)
    plt.title("原始图像+真实边界框")
    plt.axis('off')

    # 绘制预测结果（边界框+分割掩码）
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    pred_boxes = pred[0]["boxes"].cpu().numpy()
    pred_masks = pred[0]["masks"].cpu().numpy()
    # 绘制预测边界框（红色）
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = box
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-', linewidth=2)
    # 绘制分割掩码（半透明绿色）
    for mask in pred_masks:
        plt.imshow(mask[0], alpha=0.5, cmap='Greens')
    plt.title("预测结果（边界框+分割掩码）")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 随机选取3张测试图像可视化
import random

test_indices = random.sample(range(len(dataset_test)), 3)
for idx in test_indices:
    img, target = dataset_test[idx]
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(img)
    show_prediction(img.squeeze(0), target, pred)