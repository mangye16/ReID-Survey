# encoding: utf-8

import torch
from torch import nn
import collections
from .backbones.resnet import ResNet, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_nl import ResNetNL
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, GeneralizedMeanPooling, GeneralizedMeanPoolingP


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, model_name, gem_pool, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.num_classes = num_classes

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)

        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if not self.training:
            return feat

        cls_score = self.classifier(feat)
        return cls_score, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if not isinstance(param_dict, collections.OrderedDict):
            param_dict = param_dict.state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_total(score, feat, target):
            loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['total'] = criterion_total

        return criterion

