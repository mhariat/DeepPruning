from prune.pruning import *


class WeightPruner(Pruner, metaclass=ABCMeta):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_hook):
        super(WeightPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                           log_interval=log_interval, use_hook=use_hook)
        self.grad_mask = {}

    @abstractmethod
    def _compute_saliencies(self, dataloader=None):
        pass

    def _get_all_saliencies(self):
        all_saliencies= []
        for module in self.saliencies.keys():
            saliencies = self.saliencies[module]
            all_saliencies += list(saliencies[torch.gt(saliencies, 0)].detach().cpu().numpy())
        return all_saliencies

    def _make_changes(self, prune_ratio):
        threshold = self._get_threshold(prune_ratio)
        for module in self.saliencies.keys():
            mask = torch.gt(self.saliencies[module], threshold).float()
            module.weight.data *= mask

    def _update_network(self):
        for module in self.saliencies.keys():
            self.grad_mask[module] = torch.gt(torch.abs(module.weight.data), 0).float()

    def get_nb_parameters(self):
        return sum(torch.gt(torch.abs(p.data), 0).sum().item() for p in self.model.parameters())

    def get_nb_parameters_per_module(self):
        res = {}
        for k in range(len(self.interesting_modules)):
            module = self.interesting_modules[k]
            if isinstance(module, nn.Conv2d):
                key = 'Conv_{}'.format(k)
            else:
                key = 'Linear_{}'.format(k)
            res[key] = count_nonzero_parameters(module)
        return res
