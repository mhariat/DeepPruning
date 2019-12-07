from prune.channels_pruning.channels_pruning import *


class TaylorPruner(ChannelPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_out_dependencies=True):
        super(TaylorPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                           log_interval=log_interval, use_hook=True,
                                           use_out_dependencies=use_out_dependencies)
        self.gradient = OrderedDict()
        self.activation = OrderedDict()

    def _forward_func(self, module, input, output):
        super(TaylorPruner, self)._forward_func(module, input, output)
        self.activation[module] = output.data

    def _backward_func(self, module, grad_wr_input, grad_wr_output):
        super(TaylorPruner, self)._backward_func(module, grad_wr_input, grad_wr_output)
        self.gradient[module] = grad_wr_output[0]

    def _update_saliencies(self):
        for module in self.interesting_modules:
            if isinstance(module, nn.Conv2d):
                bs, cout, h, w = self.gradient[module].shape
                value = torch.abs(self.gradient[module] * self.activation[module]).sum((0, 2, 3)) / (h*w)
            else:
                value = torch.abs(self.gradient[module] * self.activation[module]).sum((0,))
            if self.normalize:
                norm = torch.sqrt((value**2)).sum()
                if 0 < norm:
                    value /= norm
            if self.steps == 0:
                self.saliencies[module] = value.cpu().numpy()
            else:
                self.saliencies[module] += value.cpu().numpy()

    def _compute_saliencies(self, dataloader=None):
        super(TaylorPruner, self)._compute_saliencies(dataloader=dataloader)
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
            self._update_saliencies()
            self.update_step()
        self.step_normalization()

    def step_normalization(self):
        for module in self.interesting_modules:
            self.saliencies[module] /= self.steps

    def _clear_buffers(self):
        super(TaylorPruner, self)._clear_buffers()
        self.gradient = OrderedDict()
        self.activation = OrderedDict()
        self.init_step()
