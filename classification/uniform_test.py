#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import numpy as np
import torch.nn as nn
from classification.utils.quantize_model import quantize_model
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    # 设置PyTorch使用的cuDNN库为非确定性模式，这通常能加速训练，但会牺牲结果的可重复性
    torch.backends.cudnn.deterministic = False
    # 允许cuDNN自动寻找最适合当前配置的高效算法，进一步加速计算
    torch.backends.cudnn.benchmark = True

    # 加载预训练全精度模型
    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    # 加载验证数据集
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet/',
                              for_inception=args.model.startswith('inception'))
    # 生成蒸馏数据，为了模拟原始数据分布的合成数据，用于量化过程中的校准
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    print('****** Data loaded ******')

    # 将单精度模型量化为8位模型，具体量化方法在quantize_model中实现
    quantized_model = quantize_model(model)
    # 冻结BatchNorm层的统计信息，因为量化过程中不希望这些统计信息发生变化
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # 使用蒸馏数据更新量化模型的激活范围
    update(quantized_model, dataloader)
    print('****** Zero Shot Quantization Finished ******')

    # 冻结激活范围
    freeze_model(quantized_model)
    # 数据并行，加速推理过程
    quantized_model = nn.DataParallel(quantized_model).cuda()

    # 测试最终量化后的模型
    test(quantized_model, test_loader)
