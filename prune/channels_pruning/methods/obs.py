from prune.channels_pruning.channels_pruning import *


class OBSPruner(ChannelPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_out_dependencies=True):
        super(OBSPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                        log_interval=log_interval, use_hook=True,
                                        use_out_dependencies=use_out_dependencies)
        self.G = OrderedDict()
        self.G_inv = OrderedDict()
        self.A = OrderedDict()

    def _forward_func(self, module, input, output):
        super(OBSPruner, self)._forward_func(module, input, output)
        if isinstance(module, nn.Conv2d):
            a = extract_patches(input[0].data, module.kernel_size, module.padding, module.stride)
        elif isinstance(module, nn.Linear):
            a = input[0].data
        else:
            raise NotImplementedError
        batch_size = a.size(0)
        a = a.view(-1, a.size(-1))
        if module.bias is not None:
            new_col = a.new_ones(a.size(0), 1)
            a = torch.cat([a, new_col], 1)
        if self.steps == 0:
            self.A[module] = (a.t() @ a)/batch_size
        else:
            self.A[module] += (a.t() @ a)/batch_size

    def _backward_func(self, module, grad_wr_input, grad_wr_output):
        super(OBSPruner, self)._backward_func(module, grad_wr_input, grad_wr_output)
        if isinstance(module, nn.Conv2d):
            g = grad_wr_output[0].transpose(1, 2).transpose(2, 3)
            spatial_size = g.size(1) * g.size(2)
        elif isinstance(module, nn.Linear):
            g = grad_wr_output[0]
            spatial_size = 1
        else:
            raise NotImplementedError
        batch_size = g.size(0)
        g = g.contiguous().view(-1, g.size(-1))
        if self.steps == 0:
            self.G[module] = (g.t() @ g)*batch_size/spatial_size
        else:
            self.G[module] += (g.t() @ g)*batch_size/spatial_size

    def _update_saliencies(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                M = weight_to_mat(module, use_patch=False)
                G_inv = get_inv(self.G[module])
                self.G_inv[module] = G_inv
                loss = (M*(M @ self.A[module])).sum(dim=1)/torch.diag(G_inv)
                if self.normalize:
                    if 0 < loss.sum():
                        loss /= loss.sum()
                self.saliencies[module] = loss.cpu().numpy()

    def _compute_saliencies(self, dataloader=None):
        super(OBSPruner, self)._compute_saliencies(dataloader=dataloader)
        self.init_step()
        for batch_idx, (data, target) in enumerate(dataloader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            if self.skip:
                output, _, _ = self.model(data)
            else:
                output = self.model(data)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()
            self.update_step()
        self.step_normalization()
        self._update_saliencies()

    def _do_surgery(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                if not hasattr(module, 'out_indices'):
                    raise NotImplementedError
                if len(module.out_indices) == 0:
                    continue
                M = weight_to_mat(module, use_patch=False)
                G_inv = self.G_inv[module]
                M = M/torch.diag(G_inv).unsqueeze(1)
                M[module.out_indices] = 0
                dM = -G_inv @ M
                dweight, dbias = mat_to_weight(module, dM)
                module.weight.data += dweight
                if module.bias is not None:
                    module.bias.data += dbias

    def _make_changes(self, prune_ratio):
        super(OBSPruner, self)._make_changes(prune_ratio=prune_ratio)
        self._do_surgery()

    def step_normalization(self):
        for module in self.interesting_modules:
            self.G[module] /= self.steps
            self.A[module] /= self.steps

    def _clear_buffers(self):
        super(OBSPruner, self)._clear_buffers()
        self.G = OrderedDict()
        self.G_inv = OrderedDict()
        self.A = OrderedDict()
        self.init_step()
