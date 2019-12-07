from prune.channels_pruning.channels_pruning import *


class NormPruner(ChannelPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_out_dependencies=True, norm='l1'):
        super(NormPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                         log_interval=log_interval, use_hook=False,
                                         use_out_dependencies=use_out_dependencies)
        self.norm = norm

    def _compute_saliencies(self, dataloader=None):
        super(NormPruner, self)._compute_saliencies(dataloader=dataloader)
        for module in self.interesting_modules:
            if isinstance(module, nn.Conv2d):
                cout, cin, kh, kw = module.weight.data.shape
                div = cin * kh * kw
                axis = (1, 2, 3)
            else:
                div = module.weight.data.shape[0]
                axis = 1
            if self.norm == 'l1':
                value = torch.abs(module.weight.data).sum(dim=axis).cpu().numpy()/div
                if self.normalize:
                    if 0 < value.sum():
                        value /= value.sum()
                self.saliencies[module] = value
            elif self.norm == 'l2':
                value = torch.sqrt((module.weight.data**2).sum(dim=axis)).cpu().numpy()/div
                if self.normalize:
                    if 0 < value.sum():
                        value /= np.sqrt((value**2).sum())
                self.saliencies[module] = value
            else:
                raise NotImplementedError
