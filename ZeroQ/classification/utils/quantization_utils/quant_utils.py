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

import math
import numpy as np
from torch.autograd import Function, Variable
import torch


def clamp(input, min, max, inplace=False):
    # 把输入Tensor的值限制在(min,max)范围内
    # 是否原地修改张量
    if inplace:
        input.clamp_(min, max)
        return input
    #所有小于min的元素被设置成min,所有大于max的元素设置为max
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    # 对于卷积层权重（4维张量），scale和zero_point需要被扩展为[out_channels, 1, 1, 1]
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    # 对于全连接层权重(2维张量)，scale和zero_point需要被扩展为[out_features,1]
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    # 量化核心计算：通过和scale相乘并且减去zero_point调整输入张量值的范围，通过四舍五入将浮点数映射到最接近的整数。
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    # 反向映射，先加零点，再除以缩放因子
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    由给定范围计算零点和缩放因子
    saturation_min: 量化范围的下界
    saturation_max: 量化范围的上界
    integral_zero_point: 是否希望零点是一个整数
    """
    # n计算了给定位数下能表示的最大整数值
    n = 2**num_bits - 1
    # scale计算了浮点数值映射到整数的缩放因子
    # clamp确保分母不为0
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    # zero_point计算了浮点数0应该映射到的整数值
    # 基于缩放因子和量化范围的下界乘积
    zero_point = scale * saturation_min
    # 零点调整为整数
    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    # 有符号量化调整，将零点调整到有符号整数能表示的中间值
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    使用给定的范围和位设置量化给定的浮点数值的类。
    只支持推理，不支持反向传播
    """
    # 静态方法，被.apply调用
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: 待量化的单精度值
        k: x的位设置
        x_min: 量化范围的下界
        x_max=None
        """
        # 如果没有提供下界和上界或者他们相等（没有有效的量化范围）
        if x_min is None or x_max is None or (sum(x_min == x_max) == 1
                                              and x_min.numel() == 1):
            #使用x的最小值和最大值作为量化范围
            x_min, x_max = x.min(), x.max()
        # 计算量化操作需要的比例因子和零点
        # TODO: asymmetric_linear_quantization_params(k, x_min, x_max)
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        # 根据得到的比例因子和零点对x量化，得到量化后的值
        # TODO: linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        # 对量化后的值进行裁剪，确保位于合法范围
        # 对于k位，合法范围是[-n, n-1]
        # 如8位，[-128,127]
        n = 2**(k - 1)
        # TODO: torch.clamp(new_quant_x, -n, n-1)裁剪是怎么做的
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        # 对裁剪后的值进行反量化，恢复到其相近的浮点表示，并将结果包装为Variable返回
        # 模拟量化操作的影响，但是保留在浮点数域中进行后续操作的能力
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
