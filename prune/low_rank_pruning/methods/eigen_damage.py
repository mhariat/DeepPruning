from prune.low_rank_pruning.low_rank_pruning import *


class EigenPruner(LowRankPruner):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_bias=True, allow_increase=False,
                 fisher_type='true'):
        super(EigenPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                          log_interval=log_interval, use_hook=True)
        self.use_bias = use_bias
        self.G = OrderedDict()
        self.A = OrderedDict()
        self.Q_G = OrderedDict()
        self.Q_A = OrderedDict()
        self.M_new_basis = OrderedDict()
        self.allow_increase = allow_increase
        self.fisher_type = fisher_type
        self.model.add_basis()

    def _forward_func(self, module, input, output):
        super(LowRankPruner, self)._forward_func(module, input, output)
        if isinstance(module, nn.Conv2d):
            a = extract_channel_patches(input[0].data, module.kernel_size, module.padding, module.stride)
            patch_size = a.size(2)
            batch_size = a.size(0)
            a = a.view(-1, a.size(-1))
            if self.use_bias & (module.bias is not None):
                new_col = a.new_ones(a.size(0), 1)/patch_size
                a = torch.cat([a, new_col], 1)
        elif isinstance(module, nn.Linear):
            a = input[0].data
            if self.use_bias & (module.bias is not None):
                new_col = a.new_ones(a.size(0), 1)
                a = torch.cat([a, new_col], 1)
            patch_size = 1
            batch_size = a.size(0)
        else:
            raise NotImplementedError
        new = (a.t() @ a)/batch_size/patch_size
        if self.steps == 0:
            self.A[module] = new
        else:
            self.A[module] += new

    def _backward_func(self, module, grad_wr_input, grad_wr_output):
        super(EigenPruner, self)._backward_func(module, grad_wr_input, grad_wr_output)
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
        new = (g.t() @ g) * batch_size / spatial_size
        if self.steps == 0:
            self.G[module] = new
        else:
            self.G[module] += new

    def _update_saliencies(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                M = weight_to_mat(module, use_patch=True)
                G = self.G[module]
                A = self.A[module]
                l_G, Q_G = get_eigen_decomp(G)
                l_A, Q_A = get_eigen_decomp(A)
                if isinstance(module, nn.Conv2d):
                    s = (l_G.unsqueeze(1) @ l_A.unsqueeze(0)).unsqueeze(1)
                elif isinstance(module, nn.Linear):
                    s = (l_G.unsqueeze(1) @ l_A.unsqueeze(0))
                else:
                    raise NotImplementedError
                M_new_basis_A = M.view(-1, M.size(-1)) @ Q_A
                M_new_basis_G = Q_G.t() @ M_new_basis_A.view(M.size(0), -1)
                M_new_basis = M_new_basis_G.view(M.size())
                saliencies_matrix = s * (M_new_basis ** 2)
                if isinstance(module, nn.Conv2d):
                    self.saliencies[module] = saliencies_matrix.sum(dim=1)
                    M_new_basis = M_new_basis.transpose(1, 2).contiguous()
                self.Q_A[module] = [Q_A]
                self.Q_G[module] = [Q_G]
                self.M_new_basis[module] = M_new_basis

    def _compute_saliencies(self, dataloader=None):
        super(EigenPruner, self)._compute_saliencies(dataloader=dataloader)
        self.init_step()
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            if self.skip:
                output, _, _ = self.model(data)
            else:
                output = self.model(data)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            if self.fisher_type == 'true':
                samples = torch.multinomial(output.data.softmax(dim=1), 1).squeeze()
                loss = criterion(output, samples)
            elif self.fisher_type == 'exp':
                loss = criterion(output, target)
            else:
                raise NotImplementedError
            loss.backward()
            self.update_step()
        self.step_normalization()
        self._update_saliencies()

    def _make_changes(self, prune_ratio):
        threshold = self._get_threshold(prune_ratio)
        for module in self.saliencies.keys():
            cout, cin = self.saliencies[module].size()
            row_saliencies = self.saliencies[module].sum(dim=1).cpu().numpy()
            col_saliencies = self.saliencies[module].sum(dim=0).cpu().numpy()
            row_indices = filter_indices(row_saliencies, threshold)
            col_indices = filter_indices(col_saliencies, threshold)
            row_ratio = 1 - len(row_indices)/cout
            col_ratio = 1 - len(col_indices)/cin
            if self.prune_ratio_limit < row_ratio:
                row_threshold = get_threshold(row_saliencies, self.prune_ratio_limit)
                row_indices = filter_indices(row_saliencies, row_threshold)
            if self.prune_ratio_limit < col_ratio:
                col_threshold = get_threshold(col_saliencies, self.prune_ratio_limit)
                col_indices = filter_indices(col_saliencies, col_threshold)
            self.M_new_basis[module] = self.M_new_basis[module][row_indices, :][:, col_indices]
            self.Q_G[module][0] = self.Q_G[module][0][:, row_indices]
            self.Q_A[module][0] = self.Q_A[module][0][:, col_indices]

    def _get_rotation_matrices(self):
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, BasisLayer):
                    if isinstance(module.basis, EigenBasisLayer):
                        sequential = module.basis.sequential
                        if isinstance(sequential[1], nn.Conv2d):
                            prev_Q_A = sequential[0].conv.weight.data
                            prev_Q_G = sequential[2].conv.weight.data
                        elif isinstance(sequential[1], nn.Linear):
                            prev_Q_A = sequential[0].linear.weight.data
                            prev_Q_G = sequential[2].linear.weight.data
                        else:
                            raise NotImplementedError
                        prev_Q_A = prev_Q_A.view(prev_Q_A.size(0), prev_Q_A.size(1)).transpose(1, 0)
                        prev_Q_G = prev_Q_G.view(prev_Q_G.size(0), prev_Q_G.size(1))
                        self.Q_A[sequential[1]].append(prev_Q_A)
                        self.Q_G[sequential[1]].append(prev_Q_G)

    def _update_rotation_matrices(self):
        with torch.no_grad():
            for module in self.interesting_modules:
                if len(self.Q_A[module]) == 2:
                    self.Q_A[module] = self.Q_A[module][1] @ self.Q_A[module][0]
                    self.Q_G[module] = self.Q_G[module][1] @ self.Q_G[module][0]
                else:
                    self.Q_A[module] = self.Q_A[module][0]
                    self.Q_G[module] = self.Q_G[module][0]

    def _update_network(self):
        self._get_rotation_matrices()
        self._update_rotation_matrices()
        self.interesting_modules = []
        for module in self.model.modules():
            if isinstance(module, BasisLayer):
                if isinstance(module.basis, EigenBasisLayer):
                    main_module = module.basis.sequential[1]
                    use_bias = self.use_bias & (module.basis.sequential[0].bias is not None)
                elif isinstance(module.basis, OriginalBasisLayer):
                    main_module = module.basis.sequential[0]
                    use_bias = self.use_bias & (main_module.bias is not None)
                else:
                    raise NotImplementedError
                Q_G = self.Q_G[main_module]
                Q_A = self.Q_A[main_module]
                M_new_basis = self.M_new_basis[main_module]
                new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias)
                new_main_module = new_basis_layer.sequential[1]
                if not self.allow_increase:
                    nb_parameters_new_basis_layer = sum([count_parameters(m) for m in expand_model(new_basis_layer)])
                    nb_parameters_prev_basis_layer = sum([count_parameters(m) for m in expand_model(module.basis)])
                    ratio = nb_parameters_new_basis_layer/nb_parameters_prev_basis_layer
                    if ratio <= 1:
                        module.basis = new_basis_layer
                        self.interesting_modules.append(new_main_module)
                    else:
                        self.interesting_modules.append(main_module)
                else:
                    module.basis = new_basis_layer
                    self.interesting_modules.append(new_main_module)

    def _get_all_saliencies(self):
        all_saliencies = []
        for module in self.saliencies.keys():
            row_saliency = self.saliencies[module].sum(dim=1)
            col_saliency = self.saliencies[module].sum(dim=0)
            if self.normalize:
                row_saliency /= row_saliency.sum()
                col_saliency /= row_saliency.sum()
            all_saliencies += list(row_saliency.cpu().numpy()) + list(col_saliency.cpu().numpy())
        return all_saliencies

    def step_normalization(self):
        for module in self.interesting_modules:
            self.G[module] /= self.steps
            self.A[module] /= self.steps

    def _clear_buffers(self):
        super(EigenPruner, self)._clear_buffers()
        self.G = OrderedDict()
        self.A = OrderedDict()
        self.Q_G = OrderedDict()
        self.Q_A = OrderedDict()
        self.M_new_basis = OrderedDict()

