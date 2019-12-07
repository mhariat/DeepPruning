from models.models import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if (stride != 1) | (inplanes != planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        is_pruned = hasattr(self.conv1, 'out_indices')

        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        if is_pruned:
            if torch.cuda.is_available():
                res = torch.cuda.FloatTensor(x.size(0), self.n_indices, out.size(2), out.size(3)).fill_(0)
            else:
                res = torch.zeros(size=(x.size(0), self.n_indices, out.size(2), out.size(3)))
            res = res.index_add(1, self.idx_out, out)
            res = res.index_add(1, self.idx_residual, residual)
            out = res
            self.additional_flops = x.size(0)*self.intersection_length*out.size(2)*out.size(3)
            self.additional_parameters =\
                self.idx_out.nelement() + self.idx_residual.nelement() - self.intersection_length + 1
        else:
            out += residual
            self.additional_flops = residual.nelement()
            self.additional_parameters = 0
        out = self.relu(out)
        return out

    def get_forward_indices(self):
        indices = list()
        indices.append(self.conv2.out_indices)
        if self.downsample is not None:
            indices.append(self.downsample[0].out_indices)
        else:
            indices.append(self.conv1.in_indices)
        all_indices = list(set(indices[0] + indices[1]))
        idx_out = []
        idx_residual = []
        n = len(all_indices)
        for i in range(n):
            if all_indices[i] in indices[0]:
                idx_out.append(i)
            if all_indices[i] in indices[1]:
                idx_residual.append(i)
        self.intersection_length = len([k for k in idx_out if k in idx_residual])
        self.idx_out = torch.tensor(idx_out)
        self.idx_residual = torch.tensor(idx_residual)
        self.n_indices = n
        if torch.cuda.is_available():
            self.idx_out = self.idx_out.cuda()
            self.idx_residual = self.idx_residual.cuda()

    def get_dependencies(self, prev_modules):
        self.conv1.dependencies = Dependencies(in_=prev_modules)
        self.bn1.dependencies = Dependencies(in_=[self.conv1])
        self.conv2.dependencies = Dependencies(in_=[self.conv1])
        self.bn2.dependencies = Dependencies(in_=[self.conv2])
        if self.downsample is not None:
            self.downsample[0].dependencies = Dependencies(in_=prev_modules)
            self.downsample[1].dependencies = Dependencies(in_=[self.downsample[0]])
            self.conv2.dependencies.update_out([self.downsample[0]])
            return [self.downsample[0], self.conv2]
        else:
            self.conv2.dependencies.update_out(prev_modules)
            return prev_modules + [self.conv2]

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        self.conv2 = BasisLayer(self.conv2)
        if self.downsample is not None:
            self.downsample[0] = BasisLayer(self.downsample[0])

    def get_additional_flops(self):
        assert hasattr(self, 'additional_flops'), 'No additional_flops attr. Consider launching forward hook!'
        return self.additional_flops

    def get_additional_parameters(self):
        assert hasattr(self, 'additional_parameters'), 'No additional_parameters attr. Consider launching forward hook!'
        return self.additional_parameters


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=1, stride=1,
                               bias=False)
        self.conv3.show = True
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if (stride != 1) | (inplanes != self.expansion*planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=self.expansion*planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        is_pruned = hasattr(self.conv1, 'out_indices')

        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        if is_pruned:
            if torch.cuda.is_available():
                res = torch.cuda.FloatTensor(x.size(0), self.n_indices, out.size(2), out.size(3)).fill_(0)
            else:
                res = torch.zeros(size=(x.size(0), self.n_indices, out.size(2), out.size(3)))
            res = res.index_add(1, self.idx_out, out)
            res = res.index_add(1, self.idx_residual, residual)
            out = res
            self.additional_flops = self.intersection_length
            self.additional_parameters =\
                self.idx_out.nelement() + self.idx_residual.nelement() - self.intersection_length + 1
        else:
            self.additional_flops = residual.nelement()
            self.additional_parameters = 0
            out += residual
        out = self.relu(out)
        return out

    def get_forward_indices(self):
        indices = list()
        indices.append(self.conv3.out_indices)
        if self.downsample is not None:
            indices.append(self.downsample[0].out_indices)
        else:
            indices.append(self.conv1.in_indices)
        all_indices = list(set(indices[0] + indices[1]))
        idx_out = []
        idx_residual = []
        n = len(all_indices)
        for i in range(n):
            if all_indices[i] in indices[0]:
                idx_out.append(i)
            if all_indices[i] in indices[1]:
                idx_residual.append(i)
        self.intersection_length = len([k for k in idx_out if k in idx_residual])
        self.idx_out = torch.tensor(idx_out)
        self.idx_residual = torch.tensor(idx_residual)
        self.n_indices = n
        if torch.cuda.is_available():
            self.idx_out = self.idx_out.cuda()
            self.idx_residual = self.idx_residual.cuda()

    def get_dependencies(self, prev_modules):
        self.conv1.dependencies = Dependencies(in_=prev_modules)
        self.bn1.dependencies = Dependencies(in_=[self.conv1])
        self.conv2.dependencies = Dependencies(in_=[self.conv1])
        self.bn2.dependencies = Dependencies(in_=[self.conv2])
        self.conv3.dependencies = Dependencies(in_=[self.conv2])
        self.bn3.dependencies = Dependencies(in_=[self.conv3])
        if self.downsample is not None:
            self.downsample[0].dependencies = Dependencies(in_=prev_modules)
            self.downsample[1].dependencies = Dependencies(in_=[self.downsample[0]])
            self.conv3.dependencies.update_out([self.downsample[0]])
            return [self.downsample[0], self.conv3]
        else:
            self.conv3.dependencies.update_out(prev_modules)
            return prev_modules + [self.conv3]

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        self.conv2 = BasisLayer(self.conv2)
        self.conv3 = BasisLayer(self.conv3)
        if self.downsample is not None:
            self.downsample[0] = BasisLayer(self.downsample[0])

    def get_additional_flops(self):
        assert hasattr(self, 'additional_flops'), 'No additional_flops attr. Consider launching forward hook!'
        return self.additional_flops

    def get_additional_parameters(self):
        assert hasattr(self, 'additional_parameters'), 'No additional_parameters attr. Consider launching forward hook!'
        return self.additional_parameters


model_layers = {
    'resnet18': (BasicBlock, [2, 2, 2, 2]),
    'resnet34': (BasicBlock, [3, 4, 6, 3]),
    'resnet50': (Bottleneck, [3, 4, 6, 3]),
    'resnet101': (Bottleneck, [3, 4, 23, 3]),
    'resnet152': (Bottleneck, [3, 8, 36, 3])
}


def get_layer_dependencies(layer, prev_modules):
    for block in layer:
        prev_modules = block.get_dependencies(prev_modules=prev_modules)
    return prev_modules


class _ResNet(Models):
    def __init__(self, depth, num_classes=1000):
        super(_ResNet, self).__init__()
        self.depth = depth
        self.name = 'Resnet'
        block, layers = model_layers['resnet{}'.format(depth)]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.fc.prunable = False
        self._initialize_weights()
        self._get_dependencies()

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = list()
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride))
        self.inplanes = planes*block.expansion
        for k in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def _get_dependencies(self):
        self.conv1.dependencies = Dependencies()
        self.bn1.dependencies = Dependencies(in_=[self.conv1])
        prev_modules = [self.conv1]
        for layer in self.layers:
            prev_modules = get_layer_dependencies(layer, prev_modules)
        self.fc.dependencies = Dependencies(in_=prev_modules)

    def prune_channels(self):
        super(_ResNet, self).prune_channels()
        for layer in self.layers:
            for block in layer:
                block.get_forward_indices()

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                module.add_basis()
        self.fc = BasisLayer(self.fc)

    def get_additional_flops(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_flops()
        return res

    def get_additional_params(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_parameters()
        return res


class _ResNet_v1(Models):
    def __init__(self, depth, num_classes=1000):
        super(_ResNet_v1, self).__init__()
        self.depth = depth
        self.name = 'Resnet'
        block, layers = model_layers['resnet{}'.format(depth)]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.fc.prunable = False
        self._initialize_weights()
        self._get_dependencies()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = list()
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride))
        self.inplanes = planes*block.expansion
        for k in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def _get_dependencies(self):
        self.conv1.dependencies = Dependencies()
        self.bn1.dependencies = Dependencies(in_=[self.conv1])
        prev_modules = [self.conv1]
        for layer in self.layers:
            prev_modules = get_layer_dependencies(layer, prev_modules)
        self.fc.dependencies = Dependencies(in_=prev_modules)

    def prune_channels(self):
        super(_ResNet_v1, self).prune_channels()
        for layer in self.layers:
            for block in layer:
                block.get_forward_indices()

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                module.add_basis()
        self.fc = BasisLayer(self.fc)

    def get_additional_flops(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_flops()
        return res

    def get_additional_params(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_parameters()
        return res


class _ResNet_v2(Models):
    def __init__(self, depth, num_classes=1000):
        super(_ResNet_v2, self).__init__()
        self.depth = depth
        assert self.depth % 6 == 2, 'Depth must be = 6n + 2!'
        self.name = 'Resnet'
        n = (depth - 2) // 6
        block, layers = (BasicBlock, [n]*3)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.fc = nn.Linear(256, num_classes)
        self.fc.prunable = False
        self._initialize_weights()
        self._get_dependencies()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.AvgPool2d(x.size()[3])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = list()
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride))
        self.inplanes = planes*block.expansion
        for k in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def _get_dependencies(self):
        self.conv1.dependencies = Dependencies()
        self.bn1.dependencies = Dependencies(in_=[self.conv1])
        prev_modules = [self.conv1]
        for layer in self.layers:
            prev_modules = get_layer_dependencies(layer, prev_modules)
        self.fc.dependencies = Dependencies(in_=prev_modules)

    def prune_channels(self):
        super(_ResNet_v2, self).prune_channels()
        for layer in self.layers:
            for block in layer:
                block.get_forward_indices()

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                module.add_basis()
        self.fc = BasisLayer(self.fc)

    def get_additional_flops(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_flops()
        return res

    def get_additional_params(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_parameters()
        return res


def _resnet(num_classes, depth):
    model = _ResNet(num_classes=num_classes, depth=depth)
    url = model_urls['resnet{}'.format(depth)]

    from torchvision.models.utils import load_state_dict_from_url
    pretrained_dict = load_state_dict_from_url(url)
    for name in pretrained_dict.copy().keys():
        if 'fc.' in name:
            pretrained_dict.pop(name)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def _resnet_v1(num_classes, depth):
    kwargs = {'num_classes': num_classes, 'depth': depth}
    return _ResNet_v1(**kwargs)


def _resnet_v2(num_classes, depth):
    kwargs = {'num_classes': num_classes, 'depth': depth}
    return _ResNet_v2(**kwargs)


def resnet(num_classes, depth, dataset=None):
    kwargs = {'num_classes': num_classes, 'depth': depth}
    if dataset == 'cifar':
        if depth % 6 == 2:
            return _resnet_v2(**kwargs)
        else:
            return _resnet_v1(**kwargs)
    else:
        return _resnet(**kwargs)

