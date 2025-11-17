import torch
from MaskRCNN import model  # 导入模型
from data.transform import data_loader, data_loader_test  # 导入数据加载器
from engine import train_one_epoch, evaluate
import torch.optim as optim

# 设置训练设备（GPU优先，无则用CPU）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 定义优化器（SGD）和学习率调度器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 每3个epoch学习率衰减10倍

# 训练10个epoch
num_epochs = 10
for epoch in range(num_epochs):
    # 训练一个epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # 更新学习率
    lr_scheduler.step()
    # 测试集评估
    evaluate(model, data_loader_test, device=device)

print("训练完成！")