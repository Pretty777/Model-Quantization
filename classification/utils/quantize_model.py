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

import torch
import torch.nn as nn
import copy
from .quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock


def quantize_model(model):
    """
    总结：递归量化预训练的单精度模型为int8量化模型。
    参数model: 预训练的单精度模型。
    """

    # Conv2d卷积层的量化和全连接层的量化
    #TODO: 分别调用了Quant_Conv2d和Quant_Linear
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=8)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=8)
        quant_mod.set_param(model)
        return quant_mod

    # 如果模型是激活层，把激活值量化为8位
    # TODO: 调用了QuantAct(activation_bit=8)
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
        return nn.Sequential(*[model, QuantAct(activation_bit=8)])

    # 如果模型是一个Sequential，递归地对每个子模块应用量化 
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m))
        return nn.Sequential(*mods)
    # 对于其他类型的模块，深拷贝模型，并递归地替换其子模块为量化模块
    else:
        # 深拷贝创建模型的一个完整副本，确保原始模型不被修改
        # 原始模型的结构和参数不会因为量化过程而改变
        q_model = copy.deepcopy(model)
        # 遍历模型的所有属性
        for attr in dir(model):
            # 获取属性对应的值或模块
            mod = getattr(model, attr)
            # 检查属性是否为nn.Module的实例，属性名不包含'norm'，以避免对标准化层进行量化
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                # 更新q_model中名为attr的子模块，将原始版本替换为量化子模块后的版本
                setattr(q_model, attr, quantize_model(mod))
        return q_model


def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model
