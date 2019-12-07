from models.models import *

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for k in range(len(cfg)):
        v = cfg[k]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    out_channels = in_channels
    return nn.Sequential(*layers), out_channels


def _vgg(num_classes=1000, depth=19, avg_pool2d=7, batch_norm=False):
    model = _VGG(num_classes=num_classes, depth=depth, avg_pool2d=avg_pool2d, batch_norm=batch_norm)
    url = model_urls['vgg{}'.format(depth)]

    from torchvision.models.utils import load_state_dict_from_url
    pretrained_dict = load_state_dict_from_url(url)
    for name in pretrained_dict.copy().keys():
        if 'classifier' in name:
            pretrained_dict.pop(name)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def vgg(num_classes=1000, depth=19, avg_pool2d=7, batch_norm=False, pretrained=False):
    kwargs = {'num_classes': num_classes, 'depth': depth, 'avg_pool2d': avg_pool2d, 'batch_norm': batch_norm}
    if pretrained:
        return _vgg(**kwargs)
    else:
        return _VGG(**kwargs)


class _VGG(Models):

    def __init__(self, num_classes=1000, depth=19, avg_pool2d=7, batch_norm=False):
        super(_VGG, self).__init__()
        self.depth = depth
        self.name = 'VGG'
        cfg = cfgs[depth]
        self.features, out_channels = make_layers(cfg=cfg, batch_norm=batch_norm)
        self.avg_pool2d = avg_pool2d
        self.avgpool = nn.AvgPool2d(avg_pool2d)
        self.classifier = nn.Linear(out_channels*avg_pool2d*avg_pool2d, num_classes)
        self.classifier.prunable = False
        self._initialize_weights()
        self._get_dependencies()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _get_dependencies(self):
        prev_modules = []
        all_modules = self.get_modules()
        for module in all_modules:
            if isinstance(module, nn.Conv2d):
                module.dependencies = Dependencies(in_=prev_modules)
                prev_modules = [module]
            elif isinstance(module, nn.Linear):
                if isinstance(prev_modules[0], nn.Conv2d):
                    module.dependencies = Dependencies(in_=prev_modules + [all_modules[all_modules.index(module) - 1]])
                else:
                    module.dependencies = Dependencies(in_=prev_modules)
                prev_modules = [module]
            elif isinstance(module, nn.BatchNorm2d):
                module.dependencies = Dependencies(in_=prev_modules)

    def _spread_dependencies(self):
        for module in self.get_module_with_dependencies():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                spread_dependencies_module(module)
            elif isinstance(module, nn.Linear):
                if isinstance(module.dependencies.in_[0], nn.Linear):
                    spread_dependencies_module(module)
                else:
                    in_indices = []
                    avg_size = self.avg_pool2d**2
                    for i in module.dependencies.in_[0].out_indices:
                        in_indices += list(np.arange(i * avg_size, i * avg_size + avg_size))
                    module.in_indices = in_indices

    def add_basis(self):
        sequential = []
        for module in self.features.modules():
            sequential.append(BasisLayer(module))
        self.features = nn.Sequential(*sequential)
        self.classifier = BasisLayer(self.classifier)