# MaskRCNN.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes, pretrained_backbone=True):
    """
    num_classes: 包括背景在内的类别数，例如 PennFudan: 2 (背景 + person)
    pretrained_backbone: 是否使用在 ImageNet/Faster-RCNN 上的预训练骨干权重
    """
    # 使用 torchvision 提供的 maskrcnn_resnet50_fpn
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained_backbone,
        progress=True,
        pretrained_backbone=pretrained_backbone
    )

    # 替换 box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换 mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
