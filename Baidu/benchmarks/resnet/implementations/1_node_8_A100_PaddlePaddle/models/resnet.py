# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D
from paddle.incubate.operators.resnet_unit import ResNetUnit
from initializer import get_param_attr

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=get_param_attr(name + '_weights', 'conv_weight'),
            bias_attr=False,
            data_format=data_format)
        if name == 'conv1':
            bn_name = 'bn_' + name
        else:
            bn_name = 'bn' + name[3:]
        self._batch_norm = BatchNorm(
            num_channels=num_filters,
            act=act,
            param_attr=get_param_attr(bn_name + '_scale', 'scale'),
            bias_attr=get_param_attr(bn_name + '_offset', 'bias'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock_old(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format='NCHW'):
        super(BottleneckBlock_old, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + '_branch2a',
            data_format=data_format)

        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + '_branch2b',
            data_format=data_format)

        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + '_branch2c',
            data_format=data_format)

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
                name=name + '_branch1',
                data_format=data_format)

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format='NCHW'):
        super(BottleneckBlock, self).__init__()

        conv_name_a = name + '_branch2a'
        bn_name_a = 'bn' + conv_name_a[3:]
        self.conv0 = ResNetUnit(
            num_channels_x=num_channels,
            num_filters=num_filters,
            filter_size=1,
            data_format=data_format,
            act='relu',
            filter_x_attr=get_param_attr(conv_name_a + '_weights',
                                         'conv_weight'),
            scale_x_attr=get_param_attr(bn_name_a + '_scale', 'scale'),
            bias_x_attr=get_param_attr(bn_name_a + '_offset', 'bias'),
            moving_mean_x_name=bn_name_a + '_mean',
            moving_var_x_name=bn_name_a + '_variance')

        conv_name_b = name + '_branch2b'
        bn_name_b = 'bn' + conv_name_b[3:]
        self.conv1 = ResNetUnit(
            num_channels_x=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            data_format=data_format,
            act='relu',
            filter_x_attr=get_param_attr(conv_name_b + '_weights',
                                         'conv_weight'),
            scale_x_attr=get_param_attr(bn_name_b + '_scale', 'scale'),
            bias_x_attr=get_param_attr(bn_name_b + '_offset', 'bias'),
            moving_mean_x_name=bn_name_b + '_mean',
            moving_var_x_name=bn_name_b + '_variance')

        conv_name_c = name + '_branch2c'
        bn_name_c = 'bn' + conv_name_c[3:]
        self.shortcut = shortcut
        if not shortcut:
            conv_z_name = name + '_branch1'
            bn_z_name = 'bn' + conv_z_name[3:]
            self.conv2 = ResNetUnit(
                num_channels_x=num_filters,
                num_filters=num_filters * 4,
                filter_size=1,
                data_format=data_format,
                act='relu',
                fuse_add=True,
                has_shortcut=True,
                filter_x_attr=get_param_attr(conv_name_c + '_weights',
                                             'conv_weight'),
                scale_x_attr=get_param_attr(bn_name_c + '_scale', 'scale'),
                bias_x_attr=get_param_attr(bn_name_c + '_offset', 'bias'),
                moving_mean_x_name=bn_name_c + '_mean',
                moving_var_x_name=bn_name_c + '_variance',
                num_channels_z=num_channels,
                stride_z=stride,
                filter_z_attr=get_param_attr(conv_z_name + '_weights',
                                             'conv_weight'),
                scale_z_attr=get_param_attr(bn_z_name + '_scale', 'scale'),
                bias_z_attr=get_param_attr(bn_z_name + '_offset', 'bias'),
                moving_mean_z_name=bn_z_name + '_mean',
                moving_var_z_name=bn_z_name + '_variance')
        else:
            self.conv2 = ResNetUnit(
                num_channels_x=num_filters,
                num_filters=num_filters * 4,
                filter_size=1,
                data_format=data_format,
                act='relu',
                fuse_add=True,
                has_shortcut=False,
                filter_x_attr=get_param_attr(conv_name_c + '_weights',
                                             'conv_weight'),
                scale_x_attr=get_param_attr(bn_name_c + '_scale', 'scale'),
                bias_x_attr=get_param_attr(bn_name_c + '_offset', 'bias'),
                moving_mean_x_name=bn_name_c + '_mean',
                moving_var_x_name=bn_name_c + '_variance')

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        y = self.conv2(conv1, inputs)

        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format='NCHW'):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + '_branch2a',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + '_branch2b',
            data_format=data_format)

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                name=name + '_branch1',
                data_format=data_format)

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


class ResNet(nn.Layer):
    def __init__(self,
                 layers=50,
                 class_dim=1000,
                 input_image_channel=3,
                 data_format='NCHW'):
        super(ResNet, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel

        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            'supported layers are {} but input layer is {}'.format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]

        num_channels = [64, 256, 512, 1024] if \
            layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name='conv1',
            data_format=self.data_format)
        self.pool2d_max = MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=self.data_format)

        self.block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name,
                            data_format=self.data_format))
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name,
                            data_format=self.data_format))
                    self.block_list.append(basic_block)
                    shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1, data_format=self.data_format)
        self.pool2d_avg_channels = num_channels[-1] * 2

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=get_param_attr('fc_0.w_0', 'fc_weight'),
            bias_attr=get_param_attr('fc_0.b_0', 'bias'))

    def forward(self, inputs):
        with paddle.static.amp.fp16_guard():
            y = self.conv(inputs)
            y = self.pool2d_max(y)
            for block in self.block_list:
                y = block(y)
            y = self.pool2d_avg(y)
            y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
            y = self.out(y)
            return y


def ResNet18(**args):
    model = ResNet(layers=18, **args)
    return model


def ResNet34(**args):
    model = ResNet(layers=34, **args)
    return model


def ResNet50(**args):
    model = ResNet(layers=50, **args)
    return model


def ResNet101(**args):
    model = ResNet(layers=101, **args)
    return model


def ResNet152(**args):
    model = ResNet(layers=152, **args)
    return model
