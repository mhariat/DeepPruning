from prune.weights_pruning.weights_pruning import *


class MagnitudePruner(WeightPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval):
        super(MagnitudePruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                              log_interval=log_interval, use_hook=False)

    def _compute_saliencies(self, dataloader=None):
        for module in self.interesting_modules:
            self.saliencies[module] = torch.abs(module.weight.data)
            if self.normalize:
                self.saliencies[module] /= torch.abs(module.weight.data).sum()
