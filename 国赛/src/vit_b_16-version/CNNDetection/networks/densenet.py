import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import Module
import torch.utils.model_zoo as model_zoo

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, BatchNorm):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', BatchNorm(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', BatchNorm(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, BatchNorm):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, BatchNorm)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, BatchNorm):
        super(_Transition, self).__init__()
        self.add_module('norm', BatchNorm(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class classification(nn.Sequential):
    def __init__(self, in_channels, out_classes, BatchNorm):
        super(classification, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.add_module('norm', BatchNorm(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.AvgPool2d(kernel_size=7, stride=1))
        self.add_module('flatten', Flatten())
        self.add_module('linear', nn.Linear(in_channels, out_classes))

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        num_layers (tuple of 4 ints) - how many layers in each pooling block  ---121-(6,12,24,16)  169-(6,12,32,32)  201-(6,12,48,32)  161-(6,12,36,24)
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        trainstion_num (int) - number of transition module  ---deleted
    """

    def __init__(self,
                 BatchNorm,
                 growth_rate=32,
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0.2,
                 num_layers=(6, 12, 24, 16),
                 transition_num=3,):

        super(DenseNet, self).__init__()

        # Low_feature 1/4 size
        self.low_feature = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),#
            ('norm0', BatchNorm(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        # self.low_feature = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),  #
        #     ('norm0', BatchNorm(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=4, stride=4))
        # ]))
        # denselyer=(6,12,24,16) densnet121
        # Middle_feature 1/16 size
        self.middle_feature = nn.Sequential()
        self.end_feature = nn.Sequential()
        num_features = num_init_features
        for i, num in enumerate(num_layers):
            if i < 2:
                bolck = _DenseBlock(num_layers=num, num_input_features=num_features,
                             bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
                num_features = num_features + num * growth_rate
                self.middle_feature.add_module('denseblock{}'.format(str(i+1)), bolck)
                if i < transition_num:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, BatchNorm=BatchNorm)
                    num_features = num_features // 2
                    self.middle_feature.add_module('transition{}'.format(str(i+1)), trans)
            else:
                bolck = _DenseBlock(num_layers=num, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
                num_features = num_features + num * growth_rate
                self.end_feature.add_module('denseblock{}'.format(str(i+1)), bolck)
                if i < transition_num:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, BatchNorm=BatchNorm)
                    num_features = num_features // 2
                    self.end_feature.add_module('transition{}'.format(str(i+1)), trans)

        #classification = classification(num_features, out_classes, BatchNorm)

        '''下面是网络结构固定的写法'''
        # num_features = num_init_features
        # block1 = _DenseBlock(num_layers=6, num_input_features=num_features,
        #                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
        # num_features = num_features + 6 * growth_rate   # 64+6*32=256
        # trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2, BatchNorm=BatchNorm)
        # num_features = num_features // 2   # 128
        # block2 = _DenseBlock(num_layers=12, num_input_features=num_features,
        #                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
        # num_features = num_features + 12 * growth_rate  # 512
        # trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2, BatchNorm=BatchNorm)
        # num_features = num_features // 2  # 256
        # self.middle_feature = nn.Sequential()
        # self.middle_feature.add_module('denseblock1', block1)
        # self.middle_feature.add_module('transition1', trans1)
        # self.middle_feature.add_module('denseblock2', block2)
        # self.middle_feature.add_module('transition2', trans2)

        # End feature 1/32 size
        # block3 = _DenseBlock(num_layers=24, num_input_features=num_features,
        #                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
        # num_features = num_features + 24 * growth_rate  # 1024
        # trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2, BatchNorm=BatchNorm)
        # num_features = num_features // 2  # 512
        # block4 = _DenseBlock(num_layers=16, num_input_features=num_features,
        #                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, BatchNorm=BatchNorm)
        # num_features = num_features + 16 * growth_rate  # 1024
        # self.end_feature = nn.Sequential(OrderedDict([
        #     ('denseblock3', block3),
        #     ('transition3', trans3),
        #     ('denseblock4', block4),
        #     ('norm', BatchNorm(bn_size * growth_rate)),
        #     ('relu', nn.ReLU(inplace=True)),
        #     ('conv', nn.Conv2d(num_features, 32, kernel_size=1, stride=1))
        # ]))


        '''参考源码的写法'''
        # # First convolution
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        # ]))
        #
        # # Each denseblock
        # num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
        #     block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
        #                         bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #     self.features.add_module('denseblock%d' % (i + 1), block)
        #     num_features = num_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        #         self.features.add_module('transition%d' % (i + 1), trans)
        #         num_features = num_features // 2
        #
        # # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        #
        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        #
        # # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        low_feature = self.low_feature(x)
        middle_feature = self.middle_feature(low_feature)
        end_feature = self.end_feature(middle_feature)
        out = F.relu(end_feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(end_feature.size(0), -1)
        return low_feature, middle_feature, end_feature, out

def densenet121(BatchNorm, pretrained=True):
    model = DenseNet(BatchNorm,
                     growth_rate=32,
                     num_init_features=64,
                     bn_size=4,
                     drop_rate=0.2,
                     num_layers=(6, 12, 24, 16),
                     transition_num=3)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['densenet121'])
        del pretrained['classifier.weight']
        del pretrained['classifier.bias']
        del pretrained['features.norm5.weight']
        del pretrained['features.norm5.bias']
        del pretrained['features.norm5.running_mean']
        del pretrained['features.norm5.running_var']
        new_state_dict = OrderedDict()
        new_state_dict2 = OrderedDict()
        blockstr = 'denseblock'
        transstr = 'transition'
        for k, v in pretrained.items():
            name = k.replace('features', 'low_feature')
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            name = k
            for i in range(4):
                if i < 2:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                else:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'end_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + '3' in name:
                        name = name.replace('low_feature', 'end_feature')
            new_state_dict2[name] = v
        model.load_state_dict(new_state_dict2)
    return model
#169-(6,12,32,32)  201-(6,12,48,32)  161-(6,12,36,24)
def densenet161(BatchNorm, pretrained=True):
    model = DenseNet(BatchNorm,
                     growth_rate=32,
                     num_init_features=64,
                     bn_size=4,
                     drop_rate=0.2,
                     num_layers=(6, 12, 36, 24),
                     transition_num=3)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['densenet161'])
        del pretrained['classifier.weight']
        del pretrained['classifier.bias']
        del pretrained['features.norm5.weight']
        del pretrained['features.norm5.bias']
        del pretrained['features.norm5.running_mean']
        del pretrained['features.norm5.running_var']
        new_state_dict = OrderedDict()
        new_state_dict2 = OrderedDict()
        blockstr = 'denseblock'
        transstr = 'transition'
        for k, v in pretrained.items():
            name = k.replace('features', 'low_feature')
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            name = k
            for i in range(4):
                if i < 2:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                else:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'end_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + '3' in name:
                        name = name.replace('low_feature', 'end_feature')
            new_state_dict2[name] = v
        model.load_state_dict(new_state_dict2)
    return model

def densenet169(BatchNorm, pretrained=True):
    model = DenseNet(BatchNorm,
                     growth_rate=32,
                     num_init_features=64,
                     bn_size=4,
                     drop_rate=0.2,
                     num_layers=(6, 12, 32, 32),
                     transition_num=3)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['densenet169'])
        del pretrained['classifier.weight']
        del pretrained['classifier.bias']
        del pretrained['features.norm5.weight']
        del pretrained['features.norm5.bias']
        del pretrained['features.norm5.running_mean']
        del pretrained['features.norm5.running_var']
        new_state_dict = OrderedDict()
        new_state_dict2 = OrderedDict()
        blockstr = 'denseblock'
        transstr = 'transition'
        for k, v in pretrained.items():
            name = k.replace('features', 'low_feature')
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            name = k
            for i in range(4):
                if i < 2:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                else:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'end_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + '3' in name:
                        name = name.replace('low_feature', 'end_feature')
            new_state_dict2[name] = v
        model.load_state_dict(new_state_dict2)#, strict=False
    return model

def densenet201(BatchNorm, pretrained=True):
    model = DenseNet(BatchNorm,
                     growth_rate=32,
                     num_init_features=64,
                     bn_size=4,
                     drop_rate=0.2,
                     num_layers=(6, 12, 48, 32),
                     transition_num=3)
    if pretrained:
        pretrained = model_zoo.load_url(model_urls['densenet201'])
        del pretrained['classifier.weight']
        del pretrained['classifier.bias']
        del pretrained['features.norm5.weight']
        del pretrained['features.norm5.bias']
        del pretrained['features.norm5.running_mean']
        del pretrained['features.norm5.running_var']
        new_state_dict = OrderedDict()
        new_state_dict2 = OrderedDict()
        blockstr = 'denseblock'
        transstr = 'transition'
        for k, v in pretrained.items():
            name = k.replace('features', 'low_feature')
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            name = k
            for i in range(4):
                if i < 2:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'middle_feature')
                else:
                    if blockstr + str(i + 1) in name:
                        name = name.replace('low_feature', 'end_feature')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('norm.', 'norm')
                    elif transstr + '3' in name:
                        name = name.replace('low_feature', 'end_feature')
            new_state_dict2[name] = v
        model.load_state_dict(new_state_dict2)
    return model

if __name__ == '__main__':
    #BachNorm =SynchronizedBatchNorm2d
    model = densenet169(BatchNorm=nn.BatchNorm2d)
    input = torch.rand(1, 3, 512, 512)
    low_feature, middle_feature, end_feature, out = model(input)
    print(low_feature.size())
    print(middle_feature.size())
    print(end_feature.size())
    print(out.size())