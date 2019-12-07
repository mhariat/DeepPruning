from models.resnet import *
from utils.data_utils import *
import torchvision.transforms as transforms


def load_checkpoint_pruning(checkpoint_path, net, use_bias):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.add_basis()
    shapes = []
    l = []
    for module_name in list(checkpoint.keys()):
        if ('sequential' in module_name) & ('bias' not in module_name):
            if '.conv.weight' not in module_name:
                n = int(module_name[-len('.weight')-1])
                l.append(n)
            shapes.append(checkpoint[module_name].shape)
    to_delete = []
    ct = 0
    for module in net.modules():
        if isinstance(module, BasisLayer):
            if l[ct] != 0:
                main_module = module.basis.sequential[0]
                use_bias_module = use_bias & (main_module.bias is not None)
                if isinstance(main_module, nn.Conv2d):
                    cout, cin, kh, kw = main_module.weight.shape
                    Q_G = torch.rand(cout, cout)
                    Q_A = torch.rand(cin, cin)
                    M_new_basis = torch.rand(cout, cin, kh, kw)
                else:
                    cout, cin = main_module.weight.shape
                    Q_G = torch.rand(cout, cout)
                    Q_A = torch.rand(cin, cin)
                    M_new_basis = torch.rand(cout, cin)
                new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias_module)
                to_delete.append(module.basis)
                module.basis = new_basis_layer
            ct += 1
    for m in to_delete:
        m.cpu()
        del m

    interesting_modules = [module for module in expand_model(net) if isinstance(module, (nn.Conv2d, nn.Linear))]
    ct = 0
    for module in interesting_modules:
        module.weight.data = torch.rand(shapes[ct])
        if isinstance(module, nn.Conv2d):
            module.out_channels = shapes[ct][0]
            module.in_channels = shapes[ct][1]
        else:
            module.out_features = shapes[ct][0]
            module.in_features = shapes[ct][1]
        ct += 1
    net.load_state_dict(checkpoint)
    return net


cifar_100 = {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}

net = resnet(100, 32, 'cifar')
root = '/home/marwane/PycharmProjects/experiments'
checkpoint_file = 'checkpoint_run_0_0.7819_0.0000.pth'
checkpoint_path = '{}/checkpoint/tucker_correct_back/{}'.format(root, checkpoint_file)
net = load_checkpoint_pruning(checkpoint_path, net, use_bias=True)


img_file = '/home/marwane/PycharmProjects/experiments/data/butterfly_s_000755.png'
img = Image.open(img_file)
img_tensor = transforms.ToTensor()(img)
img_tensor = transforms.Normalize(mean=cifar_100['mean'], std=cifar_100['std'])(img_tensor).reshape(1, 3, 32, 32)

features1 = net.layer3(net.layer2(net.layer1(net.relu(net.bn1(net.conv1(img_tensor))))))
features1 = features1.detach()
features1 = features1/features1.max()

plt.figure()
plt.imshow(img)
plt.figure()
square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features1[0, ix-1, :, :], cmap='gray')
        ix += 1
