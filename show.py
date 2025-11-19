import torch
import matplotlib
from MaskRCNN import get_instance_segmentation_model  # 导入模型构建函数
from data_transform import dataset_test  # 导入测试集
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

matplotlib.rc('font', family='SimHei')
matplotlib.rcParams['axes.unicode_minus'] = False

# 手动初始化设备（与训练保持一致）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 初始化模型并加载权重
num_classes = 2  # 背景+行人
model = get_instance_segmentation_model(num_classes)
model.to(device)
# 加载训练好的权重（替换为实际文件名，如 model_epoch9.pth）
model.load_state_dict(torch.load('maskrcnn_epoch9.pth', map_location=device))
model.eval()  # 设置为评估模式


# 以下 show_prediction 函数和可视化代码保持不变...
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
    with torch.no_grad():
        pred = model(img)
    show_prediction(img.squeeze(0), target, pred)