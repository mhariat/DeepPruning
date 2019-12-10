# Implementations of Deep Pruning methods

We implement here different pruning methods. These pruning methods have different strategies to remove parameters. Three
different type of methods are considered

- Weight pruning: Add sparsity along the neural network.
- Channel pruning: Remove entire channel dimensions resulting in whole feature maps being dropped out of the neural
 network
- Low-rank approximation pruning: Decompose a tensor into new ones with less parameters. Trade-off between weight
pruning and channel pruning as no sparsity is used but parameters are truly removed from the neural network.

Low-rank approximation methods (Tucker, CP, SVD, ...) find the decomposition by solving a minimization problem with a
specific distance. However, these algorithms don't take into account the distance distorsion caused by the uneven
loss landscape. This is a major drawback.

In the low-rank approximation bucket, one will find two different pruning methods. First, a re-implementation of the
paper [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934). The
method elaborated by this paper can be seen as a loss-oriented low-rank approximation. One will also find our improved
version of EigenDamage.

## Table Results CIFAR-100 & Resnet-32

### Test accuracy


| Methods              | 0%    | 50%   | 70%   | 80%   | 90%   |
|----------------------|-------|-------|-------|-------|-------|
| OBD                  | 78.19%| 74.01%| 72.78%| 71.93%| 67.22%|
| OBS                  | 78.19%| 74.62%| 72.97%| 72.01%| 67.53%|
| Taylor               | 78.19%| 74.50%| 73.10%| 72.69%| 67.64%|
| L1                   | 78.19%| 75.28%| 73.23%| 72.97%| 66.82%|
| Eigen-Damage         | 78.19%| 74.83%| 73.70%| 73.37%| 72.68%|
| Improved Eigen-Damage| 78.19%| **77.50%**| **76.23%**| **76.20%**| **74.93%**|

### FLOPS (in GFLOPS)

| Methods              | 0%    | 50%   | 70%   | 80%   | 90%   |
|----------------------|-------|-------|-------|-------|-------|
| Eigen-Damage         | 2.20  | 0.67  | 0.44  | 0.30  | 0.18  |
| Improved Eigen-Damage| 2.20  | **0.60**  | **0.35**  | **0.24**  | **0.14**  |


## Run

### Prerequisite:
Download CIFAR-100 dataset.


### How to run:
```
CUDA_VISIBLE_DEVICES=0 python main_prune.py --config_path config.json
```

