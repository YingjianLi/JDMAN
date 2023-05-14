import torch
import torch.nn as nn
import math
from copy import deepcopy
# 相比原版的resnet，这个可以输出feature
# 同时也进行了与训练的处理
__all__ = ['Bottleneck', 'ResNet', 'resnet18', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, args):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.args = args

        if self.args.size == 896:
            self.preconv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=False)
            self.prebn = nn.BatchNorm2d(64)
            self.conv1_ = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1_ = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, args.output_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.size == 896:
            x = self.preconv(x)
            x = self.prebn(x)
            x = self.relu(x)
            x = self.conv1_(x)
            x = self.bn1_(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.fc1(x)
        return x,feature


# class ResNet_with_ADlayer(nn.Module):
#
#     def __init__(self, block, layers, args):
#         self.inplanes = 64
#         super(ResNet_with_ADlayer, self).__init__()
#         self.args = args
#
#         if self.args.size == 896:
#             self.preconv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=False)
#             self.prebn = nn.BatchNorm2d(64)
#             self.conv1_ = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#             self.bn1_ = nn.BatchNorm2d(64)
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         # AD layers
#         self.adblock1 = nn.Sequential(
#             nn.Linear(2048,1024),
#             nn.Dropout(),
#             nn.BatchNorm1d(1024)
#         )
#
#         self.adblock2 = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.Dropout(),
#             nn.BatchNorm1d(1024)
#         )
#         self.adblock3 = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.Dropout(),
#             nn.BatchNorm1d(1024)
#         )
#         # the last layer
#         if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
#             self.fc1 = nn.Linear(1024*3, args.output_classes)
#         elif self.args.model =='resnet50_with_adlayer':
#             self.fc1 = nn.Linear(1024*3, args.output_classes) # 先每层做损失再连接做分类
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m,nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 m.bias.data.fill_(0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.args.size == 896:
#             x = self.preconv(x)
#             x = self.prebn(x)
#             x = self.relu(x)
#             x = self.conv1_(x)
#             x = self.bn1_(x)
#         else:
#             x = self.conv1(x)
#             x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         f1 = self.adblock1(x)
#         f2 = self.adblock2(f1)
#         f3 = self.adblock3(f2)
#         if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
#             features = f1
#             features = torch.cat((features,f2),1)
#             features = torch.cat((features,f3),1)
#             x = self.fc1(features)
#             return x, features
#         elif self.args.model =='resnet50_with_adlayer': # adlayer分别作为特征:
#             features = f1
#             features = torch.cat((features, f2), 1)
#             features = torch.cat((features, f3), 1)
#             x = self.fc1(features)
#             return x, f1, f2, f3

# class ResNet_with_ADlayer(nn.Module):
#
#     def __init__(self, block, layers, args):
#         self.inplanes = 64
#         super(ResNet_with_ADlayer, self).__init__()
#         self.args = args
#
#         if self.args.size == 896:
#             self.preconv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=False)
#             self.prebn = nn.BatchNorm2d(64)
#             self.conv1_ = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#             self.bn1_ = nn.BatchNorm2d(64)
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         # AD layers
#         self.adlayer1 = nn.Linear(2048,1024)
#         self.adlayer2 = nn.Linear(1024,1024)
#         self.adlayer3 = nn.Linear(1024,1024)
#         # the last layer
#         if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
#             self.fc1 = nn.Linear(1024, args.output_classes)
#         elif self.args.model =='resnet50_with_adlayer':
#             self.fc1 = nn.Linear(1024, args.output_classes) # 先每层做损失再连接做分类
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m,nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 m.bias.data.fill_(0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.args.size == 896:
#             x = self.preconv(x)
#             x = self.prebn(x)
#             x = self.relu(x)
#             x = self.conv1_(x)
#             x = self.bn1_(x)
#         else:
#             x = self.conv1(x)
#             x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         f1 = self.adlayer1(x)
#         f2 = self.adlayer2(f1)
#         f3 = self.adlayer3(f2)
#         if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
#             features = f1
#             features = torch.cat((features,f2),1)
#             features = torch.cat((features,f3),1)
#             x = self.fc1(f3) # 如果仅用最后一层会怎样？
#             return x, features
#         elif self.args.model =='resnet50_with_adlayer': # adlayer分别作为特征:
#             features = f1
#             features = torch.cat((features, f2), 1)
#             features = torch.cat((features, f3), 1)
#             x = self.fc1(features)
#             return x, f1, f2, f3
#
#             # x = self.fc1(f3)
#             # return x,f1,f2,f3


class ResNet_with_ADlayer(nn.Module):

    def __init__(self, block, layers, args):
        self.inplanes = 64
        super(ResNet_with_ADlayer, self).__init__()
        self.args = args

        if self.args.size == 896:
            self.preconv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=False)
            self.prebn = nn.BatchNorm2d(64)
            self.conv1_ = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1_ = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # # 固定以上层，仅一下层训练。
        # for p in self.parameters():
        #     p.requires_grad = False

        # AD layers
        self.adlayer1 = nn.Linear(2048, 1024)
        self.adlayer2 = nn.Linear(1024, 1024)
        self.adlayer3 = nn.Linear(1024, 1024)
        # self.adlayer1 = nn.Sequential(
        #     nn.Linear(2048,1024),
        #     nn.ReLU())
        # self.adlayer2 = nn.Sequential(
        #     nn.Linear(1024,1024),
        #     nn.ReLU())
        # self.adlayer3 = nn.Sequential(
        #     nn.Linear(1024,1024),
        #     nn.ReLU())
        # the last layer
        if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
            self.fc1 = nn.Linear(1024*3, args.output_classes)
        elif self.args.model =='resnet50_with_adlayer':
            self.fc1 = nn.Linear(1024*3, args.output_classes) # 先每层做损失再连接做分类

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.size == 896:
            x = self.preconv(x)
            x = self.prebn(x)
            x = self.relu(x)
            x = self.conv1_(x)
            x = self.bn1_(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f1 = self.adlayer1(x)
        f2 = self.adlayer2(f1)
        f3 = self.adlayer3(f2)
        if self.args.model =='resnet50_with_adlayer_all': # adlayer串联起来作为特征
            features = f1
            features = torch.cat((features,f2),1)
            features = torch.cat((features,f3),1)
            x = self.fc1(features) # 仅用最后一层进行分类
            return x, features
        elif self.args.model =='resnet50_with_adlayer': # adlayer分别作为特征:
            features = f1
            features = torch.cat((features, f2), 1)
            features = torch.cat((features, f3), 1)
            x = self.fc1(features)
            return x, f1, f2, f3

            # x = self.fc1(f3)
            # return x,f1,f2,f3

    def get_parameters(self):

        # feature
        parameter_list = [{"params": self.parameters()}]

        return parameter_list

# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, args):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.args = args
#         self.__in_features = 512*block.expansion
#
#         if self.args.size == 896:
#             self.preconv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=False)
#             self.prebn = nn.BatchNorm2d(64)
#             self.conv1_ = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
#                                     bias=False)
#             self.bn1_ = nn.BatchNorm2d(64)
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
#                                             self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
#
#         self.fc1 = nn.Linear(512 * block.expansion, args.output_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m,nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 m.bias.data.fill_(0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.feature_layers(x)
#         x = x.view(x.size(0), -1)
#         feature = x
#         x = self.fc1(x)
#         return x, feature
#
#     # def output_num(self):
#     #     return self.__in_features
#     #
#     # def get_parameters(self):
#     #
#     #     # feature
#     #     parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
#     #                               {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
#     #
#     #     return parameter_list



def resnet18(args):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = ResNet(BasicBlock, [2, 2, 2, 2], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                print(key)
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(BasicBlock, [2, 2, 2, 2], args)


def resnet50_with_adlayer(args):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = ResNet_with_ADlayer(Bottleneck, [3, 4, 6, 3], args)
        pretrained_dict = torch.load(args.pretrained)['model']
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                print(key)
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(Bottleneck, [3, 4, 6, 3], args)

def resnet50(args):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = ResNet(Bottleneck, [3, 4, 6, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                print(key)
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(Bottleneck, [3, 4, 6, 3], args)

def resnet101(args):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = ResNet(Bottleneck, [3, 4, 23, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                print(key)
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return ResNet(Bottleneck, [3, 4, 23, 3], args)

def resnet152(args):
    if args.pretrained:
        model = ResNet(Bottleneck, [3, 8, 36, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                print(key)
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
    return ResNet(Bottleneck, [3, 8, 36, 3], args)

import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
    # 实际就是李珊等人使用的目标域loss权重更新的方法

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        # nn.init.zeros_(m.bias)
        m.bias.data.fill_(0)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size,max_iter):
        super(AdversarialNetwork, self).__init__()
        self.max_iter = max_iter
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y
    def get_parameters(self):
        return [{"params": self.parameters()}]