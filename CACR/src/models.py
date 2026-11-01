import timm
import torch.nn as nn
def build(name, pretrained=True):
    m=timm.create_model(name, pretrained=pretrained, num_classes=2)
    return m
