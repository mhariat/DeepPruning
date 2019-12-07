from prune.pruning import *


class ChannelPruner(Pruner, metaclass=ABCMeta):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_hook, use_out_dependencies=True):
        super(ChannelPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                            log_interval=log_interval, use_hook=use_hook)
        self.use_out_dependencies = use_out_dependencies
        self.module_with_dependencies = model.get_module_with_dependencies()
        self._get_prunable()

    def _get_prunable(self):
        is_prunable = OrderedDict()
        for module in self.module_with_dependencies:
            is_prunable[module] = True
            if hasattr(module, 'prunable'):
                is_prunable[module] = False
            if (0 < len(module.dependencies.out_)) & self.use_out_dependencies:
                is_prunable[module] = False
        self.is_prunable = is_prunable

    def _compute_saliencies(self, dataloader=None):
        if self.use_hook:
            assert dataloader is not None, 'Must give a dataloader for this pruning method!'

    def _get_all_saliencies(self):
        all_saliencies = []
        for module in self.saliencies.keys():
            if self.is_prunable[module]:
                saliencies = list(self.saliencies[module])
                all_saliencies += saliencies
        return all_saliencies

    def _make_changes(self, prune_ratio):
        threshold = self._get_threshold(prune_ratio)
        for module in self.saliencies.keys():
            if self.is_prunable[module]:
                c_indices = filter_indices(self.saliencies[module], threshold)
                n_indices = len(c_indices)
                n_c = len(self.saliencies[module])
                ratio = 1 - n_indices/n_c
                if self.prune_ratio_limit < ratio:
                    new_threshold = get_threshold(self.saliencies[module], self.prune_ratio_limit)
                    c_indices = filter_indices(self.saliencies[module], new_threshold)
                module.out_indices = c_indices
        for module in self.saliencies.keys():
            if not self.is_prunable[module]:
                if 0 < len(module.dependencies.out_):
                    out_indices = []
                    for m_ in module.dependencies.out_:
                        out_indices += m_.out_indices
                    module.out_indices = list(set(out_indices))
                else:
                    module.out_indices = list(range(module.weight.data.shape[0]))

    def _update_network(self):
        self.model.prune_channels()

    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_nb_parameters_per_module(self):
        res = {}
        for k in range(len(self.interesting_modules)):
            module = self.interesting_modules[k]
            if isinstance(module, nn.Conv2d):
                key = 'Conv_{}'.format(k)
            else:
                key = 'Linear_{}'.format(k)
            res[key] = count_parameters(module)
        return res
